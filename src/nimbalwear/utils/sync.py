import inspect
from statistics import mean, StatisticsError

from matplotlib import pyplot as plt
from matplotlib import style as mstyle
import numpy as np
import pandas as pd
from scipy.signal import correlate
from tqdm import tqdm

mstyle.use('fast')


def sync_devices(tgt_device, ref_device, sig_labels=('Accelerometer x', 'Accelerometer y', 'Accelerometer z'),
                 last_sync=None, search_radius=None, quiet=False, **kwargs):

    """Synchronize tgt_device to ref_device based on detection of sync flips.

    Parameters
    ----------
    tgt_device
    ref_device
    sig_labels
    last_sync
    quiet
    kwargs



    """

    # get start and sample rate
    ref_start_time = ref_device.header['start_datetime']
    tgt_start_time = tgt_device.header['start_datetime']
    offset = (tgt_start_time - ref_start_time).total_seconds()

    # get accelerometer indices
    ref_sig_idx = []
    for sig_label in sig_labels:
        ref_sig_idx.append(ref_device.get_signal_index(sig_label))

    tgt_sig_idx = []
    for sig_label in sig_labels:
        tgt_sig_idx.append(tgt_device.get_signal_index(sig_label))

    ref_accel = [ref_device.signals[i] for i in ref_sig_idx]
    tgt_accel = [tgt_device.signals[i] for i in tgt_sig_idx]

    ref_freq = round(ref_device.signal_headers[ref_sig_idx[0]]['sample_rate'])
    tgt_freq = round(tgt_device.signal_headers[tgt_sig_idx[0]]['sample_rate'])

    syncs = detect_sync_flips_accel(ref_accel, tgt_accel, ref_freq, tgt_freq, offset=offset,
                                    search_radius=search_radius, **kwargs)

    # if not syncs.empty:
    #     syncs['ref_start_time'] = [ref_start_time + timedelta(seconds=(x['ref_start_idx'] / ref_freq))
    #                                for i, x in syncs.iterrows()]
    #     syncs['ref_end_time'] = [ref_start_time + timedelta(seconds=(x['ref_end_idx'] / ref_freq))
    #                              for i, x in syncs.iterrows()]
    #     syncs['tgt_start_time'] = [tgt_start_time + timedelta(seconds=(x['tgt_start_idx'] / ref_freq))
    #                                for i, x in syncs.iterrows()]
    #     syncs['tgt_end_time'] = [tgt_start_time + timedelta(seconds=(x['tgt_end_idx'] / ref_freq))
    #                              for i, x in syncs.iterrows()]

    # get sync pairs (start_times)
    ref_sync_idx = syncs['ref_start_idx'].astype(dtype=int).to_list()
    tgt_sync_idx = syncs['tgt_start_idx'].astype(dtype=int).to_list()

    # sync_pairs = [ref_sync_idx, tgt_sync_idx]
    # sync_pairs = list(map(list, zip(*sync_pairs)))

    # add "last known" if config times are withhin a couple hours ??
    #       - will be negative index at tgt config time relative to
    if last_sync is not None:
        ref_sync_idx.insert(0, round((last_sync - ref_start_time).total_seconds() * ref_freq))
        tgt_sync_idx.insert(0, round((last_sync - tgt_start_time).total_seconds() * tgt_freq))

    segments = [ref_sync_idx[:-1], ref_sync_idx[1:], tgt_sync_idx[:-1], tgt_sync_idx[1:]]
    segments = list(map(list, zip(*segments)))
    segments = pd.DataFrame(segments, columns=['ref_start_idx', 'ref_end_idx', 'tgt_start_idx', 'tgt_end_idx'])

    # get segments - calculate drift for each segment
    segments['ref_samples'] = segments['ref_end_idx'] - segments['ref_start_idx']
    segments['tgt_samples'] = segments['tgt_end_idx'] - segments['tgt_start_idx']
    segments['tgt_drift'] = segments['tgt_samples'] - (segments['ref_samples'] * (tgt_freq / ref_freq))
    segments['tgt_drift_rate'] = segments['tgt_drift'] / segments['tgt_samples']
    segments['tgt_adjust'] = [row['tgt_samples'] / row['tgt_drift'] if row['tgt_drift'] != 0 else float('nan')
                              for idx, row in segments.iterrows()]

    # correct drift by adding or deleting data points

    # loop through each signal
    for i in tqdm(range(len(tgt_device.signals)), leave=False, desc='Signals...'):

        # ratio of current signal freq to tgt signal freq (to adjust segment start and end points)
        freq_factor = tgt_device.signal_headers[i]['sample_rate'] / tgt_freq

        prev_adj = 0

        # loop through segments
        for idx, row in tqdm(segments.iterrows(), leave=False, desc='Segments...'):

            # adjust segment start and end for current signal and add or subtract adjustments from prev segments
            seg_start_idx = (0 if idx == segments.index[0] and row['tgt_start_idx'] > 0
                             else round(row['tgt_start_idx'] * freq_factor) + prev_adj)
            seg_end_idx = (len(tgt_device.signals[i]) - 1 if idx == segments.index[-1]
                           else round(row['tgt_end_idx'] * freq_factor) + prev_adj)
            seg_length = seg_end_idx - seg_start_idx
            seg_adjust_rate = abs(row['tgt_adjust'])

            if row['tgt_drift_rate'] > 0:  # if drift is positive then remove extra samples

                obj_del = []
                if seg_start_idx < 0:

                    obj_del.extend(range(round(abs(seg_start_idx) / seg_adjust_rate)))
                    obj_del.extend(
                        [round(seg_adjust_rate * (j + 1)) - 1 for j in range(round(seg_end_idx / seg_adjust_rate))])

                else:

                    obj_del.extend([seg_start_idx + round(seg_adjust_rate * (j + 1)) - 1
                                    for j in range(int(seg_length / seg_adjust_rate))])

                # delete data from each signal
                tgt_device.signals[i] = np.delete(tgt_device.signals[i], obj_del)

                prev_adj -= len(obj_del)

            elif row['tgt_drift_rate'] < 0:  # else add samples

                if seg_start_idx < 0:

                    start_insert_before = 0
                    start_insert_value = [0] * round(abs(seg_start_idx) / seg_adjust_rate)

                    insert_count = int(seg_end_idx / seg_adjust_rate)
                    insert_before = [round(seg_adjust_rate * j) for j in range(1, insert_count)]
                    insert_value = [(tgt_device.signals[i][j - 1] + tgt_device.signals[i][j]) / 2 for j in
                                    insert_before]

                    # insert data into each signal
                    tgt_device.signals[i] = np.insert(tgt_device.signals[i], insert_before, insert_value)

                    # adjust start
                    tgt_device.signals[i] = np.insert(tgt_device.signals[i], start_insert_before, start_insert_value)

                    prev_adj += len(start_insert_value) + len(insert_value)

                else:

                    insert_count = int(seg_length / seg_adjust_rate)
                    insert_before = [seg_start_idx + round(seg_adjust_rate * j) for j in range(1, insert_count)]
                    insert_value = [(tgt_device.signals[i][j - 1] + tgt_device.signals[i][j]) / 2 for j in
                                    insert_before]

                    # insert data into each signal
                    tgt_device.signals[i] = np.insert(tgt_device.signals[i], insert_before, insert_value)

                    prev_adj += len(insert_value)

    return syncs, segments


def detect_sync_flips_accel(ref_accel, tgt_accel, ref_freq, tgt_freq, offset=0, search_radius=None, **kwargs):

    ref_args = [k for k, v in inspect.signature(detect_sync_flips_ref_signal).parameters.items()]
    ref_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in ref_args}

    tgt_args = [k for k, v in inspect.signature(detect_sync_flips_tgt_signal).parameters.items()]
    tgt_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in tgt_args}

    # detect reference syncs
    ref_sync_sig_idx, ref_sync_windows = detect_sync_flips_ref_accel(ref_accel, ref_freq, **ref_dict)

    syncs = pd.DataFrame(columns=['ref_sig_idx', 'ref_start_idx', 'ref_end_idx', 'ref_flips',
                                  'ref_ae', 'tgt_sig_idx', 'tgt_start_idx', 'tgt_end_idx',
                                  'tgt_corr'])

    for s in tqdm(ref_sync_windows[0], desc="Searching target for detected sync flips...", leave=False):

        ref_sync = ref_accel[ref_sync_sig_idx][s[0]:s[1]]

        if search_radius is not None:
            mid_sync_ref = s[0] + ((s[1] - s[0]) / 2)
            sample_gain = tgt_freq / ref_freq
            sample_offset = int(offset * tgt_freq)
            mid_sync_tgt = int(mid_sync_ref * sample_gain - sample_offset)
            sample_radius = int(search_radius * tgt_freq)

            start_i = mid_sync_tgt - sample_radius
            start_i = 0 if start_i < 0 else start_i
            start_i = len(tgt_accel[0]) - 1 if start_i >= len(tgt_accel[0]) else start_i

            end_i = mid_sync_tgt + sample_radius
            end_i = 0 if end_i < 0 else end_i
            end_i = len(tgt_accel[0]) - 1 if end_i >= len(tgt_accel[0]) else end_i

            tgt_accel_window = [a[start_i:end_i] for a in tgt_accel]

        else:
            start_i = 0
            tgt_accel_window = tgt_accel

        if start_i != end_i:

            tgt_sync_sig_idx, tgt_sync = detect_sync_flips_tgt_accel(tgt_accel_window, ref_sync, tgt_freq, ref_freq,
                                                                     **tgt_dict)

            new_sync = pd.Series([int(ref_sync_sig_idx), int(s[0]), int(s[1]), int(s[2]), round(s[3], 3),
                                  int(tgt_sync_sig_idx), int(tgt_sync[0] + start_i), int(tgt_sync[1] + start_i),
                                  round(tgt_sync[2] ,2)],
                                 index=syncs.columns)

            syncs = syncs.append(new_sync, ignore_index=True)

    return syncs


def detect_sync_flips_ref_accel(accel, signal_freq, **kwargs):

    ref_sync_sig_idx = None
    ref_sync_windows = None
    prev_axis_mean_ae = 1
    r_i = 0
    for ref_signal in tqdm(accel, desc="Searching reference axes for sync flips...", leave=False):

        # detect reference sync
        axis_sync_windows = detect_sync_flips_ref_signal(ref_signal, signal_freq, **kwargs)

        try:
            axis_mean_ae = mean([asw[3] for asw in axis_sync_windows[0]])
        except StatisticsError:
            r_i += 1
            continue

        # only get sync windows from axis with best match
        if axis_mean_ae < prev_axis_mean_ae:
            prev_axis_mean_ae = axis_mean_ae
            ref_sync_sig_idx = r_i
            ref_sync_windows = axis_sync_windows

        r_i += 1

    return ref_sync_sig_idx, ref_sync_windows


def detect_sync_flips_ref_signal(signal, signal_freq, signal_ds=1, rest_min=2, rest_max=15, rest_sens=0.12, flip_max=2,
                                 min_flips=4, reject_above_ae=0.2, plot_detect_ref=False, plot_quality_ref=False):

    """Detect sync flips in a single reference signal by searching for "flips" between periods of rest

    """

    # downsample
    signal = signal[::signal_ds]

    ##############################################
    # detect periods of rest (device is resting)
    ##############################################

    # calculate min and max length of rest period in datapoints
    rest_min_dp = rest_min * signal_freq / signal_ds
    rest_max_dp = rest_max * signal_freq / signal_ds

    i = 0
    rest_ind = []  # contains tuple with start and end indices of each rest period

    # loop through signal looking for periods where device isn't moving
    while i < (len(signal) - rest_min_dp):
        j = i + 1
        while (abs(signal[j] - signal[i]) <= (abs(signal[i]) * rest_sens)) & (j < (len(signal) - 1)):
            j += 1
        if rest_min_dp <= (j - i) <= rest_max_dp:
            rest_ind.append((i, j - 1))
        i = j

    ###############################################
    # detect flips (device orientation is flipped)
    ###############################################

    # caclulate max duration of flip in datapoints
    flip_max_dp = round(flip_max * signal_freq / signal_ds)

    flip_ind = []

    # loop through periods of rest and only keep those with a flip between
    for k in range(len(rest_ind) - 1):
        if (((rest_ind[k + 1][0] - rest_ind[k][1]) <= flip_max_dp) &
                ((signal[rest_ind[k + 1][0]] * signal[rest_ind[k][0]]) < 0)):
            flip_ind.append(rest_ind[k])
            flip_ind.append(rest_ind[k + 1])

    flip_ind = sorted(set(flip_ind), key=lambda y: y[0])

    ########################################
    # detect sync windows (series of flips)
    ########################################
    sync_ind = []

    i_start = 0
    i_end = 0
    flips = 0

    for i in flip_ind:
        if (i[0] - i_end) > flip_max_dp:
            if flips >= min_flips:
                sync_ind.append((i_start, i_end, flips))
            flips = 0
            i_start = i[0]
            i_end = i[1]
        else:
            flips += 1
            i_end = i[1]

    if flips >= min_flips:
        sync_ind.append((i_start, i_end, flips))

    ########################################
    # Calculate sync signal quality
    ########################################

    # based on mean absolute error of sync signal from gravity

    keep_sync_ind = []
    rej_sync_ind = []

    for s in sync_ind:

        sync_signal = [abs(x) for x in signal[s[0]:s[1]]]
        abs_err = round(mean([abs(x - 1) for x in sync_signal]), 3)

        if abs_err <= reject_above_ae:
            keep_sync_ind.append((s[0], s[1], s[2], abs_err))
        else:
            rej_sync_ind.append((s[0], s[1], s[2], abs_err))

    ###########
    # Plot
    ###########

    if plot_detect_ref:

        signal_linewidth = 0.25
        signal_color = 'grey'

        fig, ax = plt.subplots(3, 1, sharex='all', sharey='all', figsize=(15, 9))

        ax[0].set_title('Rest periods')
        ax[0].plot(signal, linewidth=signal_linewidth, color=signal_color)

        y_lim = max(max(signal), abs(min(signal)))
        ax[0].set_ylim(-y_lim, y_lim)

        for x in rest_ind:
            if signal[x[0]] >= 0:
                ax[0].axvspan(xmin=x[0], xmax=x[1], ymin=0.15, ymax=0.85, alpha=0.5, color='cornflowerblue')
            else:
                ax[0].axvspan(xmin=x[0], xmax=x[1], ymin=0.15, ymax=0.85, alpha=0.5, color='gold')

        ax[1].set_title('Rest periods with flips')
        ax[1].plot(signal, linewidth=signal_linewidth, color=signal_color)

        for x in flip_ind:
            if signal[x[0]] >= 0:
                ax[1].axvspan(xmin=x[0], xmax=x[1], ymin=0.15, ymax=0.85, alpha=0.5, color='cornflowerblue')
            else:
                ax[1].axvspan(xmin=x[0], xmax=x[1], ymin=0.15, ymax=0.85, alpha=0.5, color='gold')

        ax[2].set_title('Sync windows')
        ax[2].plot(signal, linewidth=signal_linewidth, color=signal_color)

        for x in flip_ind:
            if signal[x[0]] >= 0:
                ax[2].axvspan(xmin=x[0], xmax=x[1], ymin=0.15, ymax=0.85, alpha=0.2, color='cornflowerblue')
            else:
                ax[2].axvspan(xmin=x[0], xmax=x[1], ymin=0.15, ymax=0.85, alpha=0.2, color='gold')

        for x in keep_sync_ind:
            ax[2].axvspan(xmin=x[0], xmax=x[1], ymin=0, ymax=1, alpha=0.5, color='palegreen')

        for x in rej_sync_ind:
            ax[2].axvspan(xmin=x[0], xmax=x[1], ymin=0, ymax=1, alpha=0.5, color='lightcoral')

    if plot_quality_ref:

        signal_linewidth = 0.25
        accepted_color = 'green'
        rejected_color = 'red'

        nrows = max(len(keep_sync_ind), len(rej_sync_ind))

        if nrows > 0:

            # setup figure
            fig, axs = plt.subplots(nrows, 2, sharey='all', squeeze=False, figsize=(10, 9))
            fig.suptitle("Sync signals")
            axs[0][0].set_title("Accepted", color=accepted_color)
            axs[0][1].set_title("Rejected", color=rejected_color)

            # plot accepted
            i = 0
            for w in keep_sync_ind:
                sync_signal = [abs(x) for x in signal[w[0]:w[1]]]
                axs[i][0].plot(sync_signal, linewidth=signal_linewidth, color=accepted_color)
                axs[i][0].set_ylim(0, 2)
                axs[i][0].text(0.1, 1.6, f"AE = {w[3]}", size=15, color=accepted_color)
                i += 1

            # plot rejected
            i = 0
            for w in rej_sync_ind:
                sync_signal = [abs(x) for x in signal[w[0]:w[1]]]
                axs[i][1].plot(sync_signal, linewidth=signal_linewidth, color=rejected_color)
                axs[i][1].set_ylim(0, 2)
                axs[i][1].text(0.1, 1.6, f"AE = {w[3]}", size=15, color=rejected_color)
                i += 1

    accepted_sync_flips = [(round(x[0] * signal_ds), round(x[1] * signal_ds), x[2], x[3]) for x in keep_sync_ind]
    rejected_sync_flips = [(round(x[0] * signal_ds), round(x[1] * signal_ds), x[2], x[3]) for x in rej_sync_ind]

    return accepted_sync_flips, rejected_sync_flips


def detect_sync_flips_tgt_accel(tgt_accel, ref_sync, tgt_freq, ref_freq, **kwargs):

    tgt_sync_sig_idx = None
    tgt_sync = (None, None, None)

    prev_tgt_corr = 0
    t_i = 0
    for tgt_signal in tqdm(tgt_accel, desc="Searching target axes for sync flips...", leave=False):

        axis_tgt_sync = detect_sync_flips_tgt_signal(tgt_signal, ref_sync, tgt_freq, ref_freq, **kwargs)

        axis_tgt_corr = axis_tgt_sync[2]

        if axis_tgt_corr > prev_tgt_corr:
            prev_tgt_corr = axis_tgt_corr
            tgt_sync_sig_idx = t_i
            tgt_sync = axis_tgt_sync

        t_i += 1

    return tgt_sync_sig_idx, tgt_sync


def detect_sync_flips_tgt_signal(tgt_signal, ref_sync, tgt_signal_freq, ref_signal_freq, plot_detect_tgt=False):

    """Detect a reference sync flip within a single target signal by finding the highest cross-correlation.

    """

    sync_start = sync_end = sync_corr = None

    # ref_ds = 1
    tgt_ds = 1

    if ref_signal_freq > tgt_signal_freq:
        ref_ds = round(ref_signal_freq / tgt_signal_freq)
        ref_sync = ref_sync[::ref_ds]
    elif tgt_signal_freq > ref_signal_freq:
        tgt_ds = round(tgt_signal_freq / ref_signal_freq)
        tgt_signal = tgt_signal[::tgt_ds]

    # corr = correlate(tgt_signal, ref_sync) / len(ref_sync)
    if tgt_signal is not None:
        corr = correlate(tgt_signal, ref_sync) / max(correlate(ref_sync, ref_sync))

        max_corr_idx = np.argmax(abs(corr))
        max_corr = round(corr[max_corr_idx], 2)

        tgt_sync_end = max_corr_idx + 1
        tgt_sync_start = tgt_sync_end - len(ref_sync)

        if plot_detect_tgt:
            plt.figure()
            plt.title(f'corr = {max_corr}')
            win_start = max(0, tgt_sync_start - round(len(ref_sync) / 2))
            win_end = min(tgt_signal.shape[0] - 1, tgt_sync_end + round(len(ref_sync) / 2))
            plt.plot(range(win_start, win_end),
                     tgt_signal[win_start:win_end],
                     color='black')
            if max_corr < 0:
                ref_sync = [-x for x in ref_sync]
            plt.plot(range(tgt_sync_start, tgt_sync_end), ref_sync, color='red')
            plt.show()

        sync_start = round(tgt_sync_start * tgt_ds)
        sync_end = round(tgt_sync_end * tgt_ds)
        sync_corr = round(abs(max_corr), 2)

    return sync_start, sync_end, sync_corr
