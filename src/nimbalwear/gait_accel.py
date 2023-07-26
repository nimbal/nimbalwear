import os
from datetime import timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths


def window_correlate(sig1, sig2):
    """
    Does cross-correlation between 2 signals over a window of indices
    """
    sig1 = sig1 if type(sig1) is np.ndarray else np.array(sig1)
    sig2 = sig2 if type(sig2) is np.ndarray else np.array(sig2)

    sig = max([sig1, sig2], key=len)
    window = min([sig1, sig2], key=len)

    engine = 'cython' if len(sig) < 100000 else 'numba'
    cc = pd.Series(sig).rolling(window=len(window)).apply(lambda x: np.corrcoef(x, window)[0, 1], raw=True,
                                                          engine=engine).shift(-len(window) + 1).fillna(0).to_numpy()

    return cc


def push_off_detection(vert_accel, pushoff_avg, freq, pushoff_threshold=0.85):
    """
    Detects the steps based on the pushoff_df, uses window correlate and cc threshold  to accept/reject pushoffs
    """

    cc_list = window_correlate(vert_accel, pushoff_avg)

    # TODO: Postponed -- DISTANCE CAN BE ADJUSTED FOR THE LENGTH OF ONE STEP RIGHT NOW ASSUMPTION IS THAT A PERSON
    # CANT TAKE 2 STEPS WITHIN 0.5s
    pushoff_ind, _ = find_peaks(cc_list, height=pushoff_threshold, distance=max(0.2 * freq, 1))

    return pushoff_ind


def mid_swing_peak_detect(data, pushoff_ind, freq, swing_phase_time=0.3):
    """
    Detects a peak within the swing_detect window length - swing peak
    """

    swing_detect_len = int(freq * swing_phase_time)  # length to check for swing
    detect_window = data[pushoff_ind:pushoff_ind + swing_detect_len]
    peaks, prop = find_peaks(-detect_window,
                             distance=max(swing_detect_len * 0.25, 1),
                             prominence=0.2, wlen=swing_detect_len,
                             width=[0 * freq, swing_phase_time * freq], rel_height=0.75)
    if len(peaks) == 0:
        return None

    results = peak_widths(-detect_window, peaks)
    prop['widths'] = results[0]

    return pushoff_ind + peaks[np.argmax(prop['widths'])]


def swing_detect(data, pushoff_ind, mid_swing_ind, freq, swing_up_detect_time=0.1):
    """
    Detects swings (either up or down) given a starting index (window_ind).
    Swing duration is preset - currently unused and mid_swing_peak_detect is used in place of this function
    """
    # swinging down
    detect_window = data[pushoff_ind:mid_swing_ind]
    swing_len = mid_swing_ind - pushoff_ind
    swing_down_sig = -np.arange(swing_len) + swing_len / 2 + np.mean(detect_window)

    # swinging up
    swing_up_detect = int(freq * swing_up_detect_time)  # length to check for swing
    swing_up_detect_window = data[mid_swing_ind:mid_swing_ind + swing_up_detect]
    swing_up_sig = -(-np.arange(swing_up_detect) + swing_up_detect / 2 + np.mean(detect_window))

    swing_down_cc = [np.corrcoef(detect_window, swing_down_sig)[0, 1]] if detect_window.shape[0] > 1 else [
        0]
    swing_up_cc = [np.corrcoef(swing_up_detect_window, swing_up_sig)[0, 1]] if swing_up_detect_window.shape[
                                                                                   0] > 1 else [0]

    return swing_down_cc, swing_up_cc


def heel_strike_detect(data, window_ind, freq, heel_strike_detect_time=0.5):
    """
    Detects a heel strike based on the change in acceleration over time (first derivative).
    """
    heel_detect = int(freq * heel_strike_detect_time)
    detect_window = data[window_ind:window_ind + heel_detect]
    accel_t_plus1 = np.append(
        detect_window[1:detect_window.size], detect_window[-1])
    accel_t_minus1 = np.insert(detect_window[:-1], 0, detect_window[0])
    accel_derivative = (accel_t_plus1 - accel_t_minus1) / (2 / freq)

    return accel_derivative


def detect_steps(vert_accel, freq, pushoff_df, pushoff_threshold=0.85, pushoff_time=0.4,
                 swing_phase_time=0.3, heel_strike_detect_time=0.5,
                 heel_strike_threshold=-5, foot_down_time=0.05):

    # TODO: adjust freq of pushoff_df ??

    pushoff_len = int(pushoff_time * freq)

    print("Pushoff Detection...")
    pushoff_ind = push_off_detection(vert_accel, pushoff_df['avg'], freq, pushoff_threshold=pushoff_threshold)
    end_pushoff_ind = pushoff_ind + pushoff_len
    state_arr = np.zeros(vert_accel.size)

    # dict of potential detected errors
    detects = {'push_offs': len(end_pushoff_ind),   # number of pushoffs
               'mid_swing_peak': [],                # no mid swing peak detected
               'swing_up': [],                      # not used?
               'swing_down': [],                    # not used?
               'heel_strike': [],                   # no heel strike detected
               'next_i': [],                        # next i too close to previous i
               'pushoff_mean': []}                  # pushoff means not within 1 std of model means
    detect_arr = np.zeros(vert_accel.size)

    # initialize
    end_i = None
    step_indices = []
    step_lengths = []

    # run
    for count, i in tqdm(enumerate(end_pushoff_ind), total=len(end_pushoff_ind), leave=False, desc='Step Detection'):
        # check if next index within the previous detection
        if end_i and i - pushoff_len < end_i:
            detects['next_i'].append(i - 1)
            continue

        # mean/std check for pushoff, state = 1
        pushoff_mean = np.mean(vert_accel[i - pushoff_len:i])
        upper = (pushoff_df['avg'] + pushoff_df['std'])
        lower = (pushoff_df['avg'] - pushoff_df['std'])
        if not np.any((pushoff_mean < upper) & (pushoff_mean > lower)):
            detects['pushoff_mean'].append(i - 1)
            continue

        mid_swing_i = mid_swing_peak_detect(vert_accel, i, freq, swing_phase_time=swing_phase_time)
        if mid_swing_i is None:
            detects['mid_swing_peak'].append(i - 1)
            continue

        accel_derivatives = heel_strike_detect(vert_accel, mid_swing_i, freq,
                                               heel_strike_detect_time=heel_strike_detect_time)
        accel_threshold_list = np.where(
            accel_derivatives < heel_strike_threshold)[0]
        if len(accel_threshold_list) == 0:
            detects['heel_strike'].append(i - 1)
            continue
        accel_ind = accel_threshold_list[0] + mid_swing_i
        end_i = accel_ind + int(foot_down_time * freq)

        state_arr[i - pushoff_len:i] = 1
        state_arr[i:mid_swing_i] = 2
        state_arr[mid_swing_i:accel_ind] = 3
        state_arr[accel_ind:end_i] = 4

        step_indices.append(i - pushoff_len)
        step_lengths.append(end_i - (i - pushoff_len))

    detect_arr[detects['swing_down']] = 1
    detect_arr[detects['swing_up']] = 2
    detect_arr[detects['heel_strike']] = 3
    detect_arr[detects['next_i']] = 4
    detect_arr[detects['pushoff_mean']] = 5
    detect_arr[detects['mid_swing_peak']] = 6

    return state_arr, detect_arr, step_indices, step_lengths


def export_steps(vert_accel, detect_arr, state_arr, freq, start_time, step_indices, pushoff_time=0.4,
                 foot_down_time=0.05, success=True):
    """
    Export steps into a dataframe -  includes all potential push-offs and the state that they fail on
    """
    timestamps = pd.date_range(start=start_time, periods=len(vert_accel), freq=f"{round(1 / freq, 6)}S")

    assert len(detect_arr) == len(timestamps)
    failed_step_indices = np.where(detect_arr > 0)[0]
    failed_step_timestamps = timestamps[failed_step_indices]

    error_mapping = {1: 'swing_down',
                     2: 'swing_up',
                     3: 'heel_strike_too_small',
                     4: 'too_close_to_next_i',
                     5: 'too_far_from_pushoff_mean',
                     6: 'mid_swing_peak_not_detected'}
    failed_step_state = list(map(error_mapping.get, detect_arr[failed_step_indices]))

    step_timestamps = timestamps[step_indices]

    swing_start = np.where((state_arr == 1) & (np.roll(state_arr, -1) == 2))[0]
    mid_swing = np.where((state_arr == 2) & (np.roll(state_arr, -1) == 3))[0]
    heel_strike = np.where((state_arr == 3) & (np.roll(state_arr, -1) == 4))[0]

    pushoff_start = swing_start - int(pushoff_time * freq)
    gait_cycle_end = heel_strike + int(foot_down_time * freq)
    step_durations = (gait_cycle_end - pushoff_start) / freq

    assert len(step_indices) == len(swing_start)
    assert len(step_indices) == len(mid_swing)
    assert len(step_indices) == len(heel_strike)

    successful_steps = pd.DataFrame({
        'step_timestamp': step_timestamps,
        'step_index': np.array(step_indices),
        'step_state': 'success',
        'swing_start_time': timestamps[swing_start],
        'mid_swing_time': timestamps[mid_swing],
        'heel_strike_time': timestamps[heel_strike],
        'swing_start_accel': vert_accel[swing_start],
        'mid_swing_accel': vert_accel[mid_swing],
        'heel_strike_accel': vert_accel[heel_strike],
        'step_duration': step_durations,
    })
    failed_steps = pd.DataFrame({
        'step_time': failed_step_timestamps,
        'step_index': np.array(failed_step_indices),
        'step_state': failed_step_state
    })
    if success:
        df = successful_steps
    else:
        df = pd.concat([successful_steps, failed_steps], sort=True)
        df = df.sort_values(by='step_index')
        df = df.reset_index(drop=True)

    return df


def create_pushoff_model(vert_accel, freq, step_indices, pushoff_time=0.4, peaks=None, quiet=False):
    """
    Creates average pushoff dataframe that is used to find pushoff data
    """
    pushoff_sig_list = []
    pushoff_len = pushoff_time * freq

    if not peaks:
        peaks = np.array(step_indices) + pushoff_len
    for i, peak_ind in tqdm(enumerate(peaks), desc="Generating pushoff average", total=len(peaks), disable=quiet,
                            leave=False):
        pushoff_sig = vert_accel[int(peak_ind - pushoff_len):int(peak_ind)]
        pushoff_sig_list.append(pushoff_sig)

    pushoff_sig_list = np.array(pushoff_sig_list)

    po_avg_sig = np.mean(pushoff_sig_list, axis=0)
    po_std_sig = np.std(pushoff_sig_list, axis=0)
    po_max_sig = np.max(pushoff_sig_list, axis=0)
    po_min_sig = np.min(pushoff_sig_list, axis=0)

    pushoff_df = pd.DataFrame(
        {'avg': po_avg_sig, 'std': po_std_sig, 'max': po_max_sig, 'min': po_min_sig})

    return pushoff_df


def calc_step_parameters(vert_accel, steps_df, freq, pushoff_time=0.4, heel_strike_detect_time=0.5):

    toe_offs = steps_df.loc[steps_df['step_state'] == 'success', 'swing_start_time']
    mid_swings = steps_df.loc[steps_df['step_state'] == 'success', 'mid_swing_time']
    heel_strikes = steps_df.loc[steps_df['step_state'] == 'success', 'heel_strike_time']
    step_indices = steps_df.loc[steps_df['step_state'] == 'success', 'step_index']

    mid_swing_indices = step_indices + (pushoff_time + (mid_swings - toe_offs).dt.total_seconds()) * freq

    swingdown_times = (mid_swings - toe_offs).dt.total_seconds()
    swingup_times = (heel_strikes - mid_swings).dt.total_seconds()
    heelstrike_values = [np.min(heel_strike_detect(vert_accel, int(ms_ind), freq, heel_strike_detect_time))
                         for ms_ind in mid_swing_indices]

    swing_down_mean = np.nanmean(swingdown_times)
    swing_down_std = np.nanstd(swingdown_times)
    swing_up_mean = np.nanmean(swingup_times)
    swing_up_std = np.nanstd(swingup_times)
    heel_strike_mean = np.nanmean(sorted(heelstrike_values, reverse=True)[:len(heelstrike_values) // 4])
    heel_strike_std = np.nanstd(sorted(heelstrike_values, reverse=True)[:len(heelstrike_values) // 4])

    return swing_down_mean, swing_down_std, swing_up_mean, swing_up_std, heel_strike_mean, heel_strike_std


def calc_detection_parameters(vert_accel, freq, step_indices, steps_df, pushoff_time=0.4, heel_strike_detect_time=0.5,
                              peaks=None):

    pushoff_df = create_pushoff_model(vert_accel, freq, step_indices, pushoff_time=pushoff_time, peaks=peaks)

    par = calc_step_parameters(vert_accel, steps_df, freq, pushoff_time=pushoff_time,
                               heel_strike_detect_time=heel_strike_detect_time)

    swing_down_mean, swing_down_std, swing_up_mean, swing_up_std, heel_strike_mean, heel_strike_std = par

    swing_phase_time = swing_down_mean + swing_down_std + swing_up_mean + swing_up_std
    swing_phase_time = max(swing_phase_time, 0.1)
    heel_strike_detect_time = 0.5 + swing_up_mean + 2 * swing_up_std
    heel_strike_threshold = -3 - heel_strike_mean / (2 * heel_strike_std)
    # TODO: confirm this should be heel_strike_std, was heel_strike_threshold

    return pushoff_df, swing_phase_time, heel_strike_detect_time, heel_strike_threshold


def state_space_accel_steps(vert_accel, freq, start_time, pushoff_df=None, pushoff_threshold=0.85,
                            pushoff_time=0.4, swing_down_detect_time=0.1, swing_up_detect_time=0.1,
                            heel_strike_detect_time=0.5, heel_strike_threshold=-5, foot_down_time=0.05, success=True,
                            update_pars=True, return_default=False):

    """
    Detects the steps within the accelerometer data. Based on this paper:
    https://ris.utwente.nl/ws/portalfiles/portal/6643607/00064463.pdf
    ---
    Parameters
    ---
    vert_accel -> vertical axis accelerometer data
    start_dp, end_dp -> indexed for start of step and end of step detection
    axis -> axis for vertical acceleration; default None but uses output from "get_acc_data_ssc"
    pushoff_df -> dataframe for pushoff detect; default is True to import premade pushoff df
    timestamps -> timestamps for data from "get_acc_data_ssc"
    ---
    Return
    ---
    steps_df -> dataframe with indexes of steps detected (beginning of step) from ssc algorithm
    """

    swing_phase_time = swing_down_detect_time + swing_up_detect_time * 2  # TODO: confirm order-of-operations

    # set initial pushoff df - if none load default
    if pushoff_df is None:  # importing static pushoff_df
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pushoff_df = pd.read_csv(os.path.join(dir_path, 'data', 'pushoff_df.csv'))

    # detect steps with passed or default parameters
    step_out = detect_steps(vert_accel=vert_accel, freq=freq, pushoff_df=pushoff_df,
                            pushoff_threshold=pushoff_threshold, pushoff_time=pushoff_time,
                            swing_phase_time=swing_phase_time, heel_strike_detect_time=heel_strike_detect_time,
                            heel_strike_threshold=heel_strike_threshold, foot_down_time=foot_down_time)

    state_arr, detect_arr, step_indices, step_lengths = step_out

    # export steps
    default_steps_df = export_steps(vert_accel, detect_arr, state_arr, freq, start_time, step_indices,
                                    pushoff_time=pushoff_time, foot_down_time=foot_down_time, success=success)

    if update_pars:

        # check for enough steps and check for None before updating parameters and re-running
        if len(step_indices) >= 20:

            # update detection parameters
            par = calc_detection_parameters(vert_accel, freq, step_indices, default_steps_df, pushoff_time=pushoff_time,
                                            heel_strike_detect_time=heel_strike_detect_time)

            pushoff_df, swing_phase_time, heel_strike_detect_time, heel_strike_threshold = par

            # detect steps with updated parameters
            step_out = detect_steps(vert_accel=vert_accel, freq=freq, pushoff_df=pushoff_df,
                                    pushoff_threshold=pushoff_threshold, pushoff_time=pushoff_time,
                                    swing_phase_time=swing_phase_time, heel_strike_detect_time=heel_strike_detect_time,
                                    heel_strike_threshold=heel_strike_threshold, foot_down_time=foot_down_time)

            state_arr, detect_arr, step_indices, step_lengths = step_out

            # export steps
            steps_df = export_steps(vert_accel, detect_arr, state_arr, freq, start_time, step_indices,
                                    pushoff_time=pushoff_time, foot_down_time=foot_down_time, success=success)

        else:

            print("Fewer than 20 steps detected. Could not update detection model.")
            steps_df = default_steps_df

    else:
        steps_df = default_steps_df

    if return_default:
        return steps_df, default_steps_df
    else:
        return steps_df


if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    from src.nimbalwear import Device

    show_plots=True

    # GNAC testing
    ankle_path = Path("W:/NiMBaLWEAR/OND06/processed/standard_device_edf/GNAC/OND06_1027_01_GNAC_LAnkle.edf")
    ankle = Device()
    ankle.import_edf(ankle_path)

    # get signal idxs
    y_idx = ankle.get_signal_index('Accelerometer y')

    # get signal frequencies needed for step detection
    fs = ankle.signal_headers[y_idx]['sample_rate']

    vertical_acc = ankle.signals[y_idx]

    data_start_time = ankle.header['start_datetime']

    steps_df, default_steps_df = state_space_accel_steps(vert_accel=vertical_acc, freq=fs, start_time=data_start_time,
                                                         update_pars=True, return_default=True)


    timestamps = pd.date_range(start=data_start_time, periods=len(vertical_acc), freq=f"{round(1 / fs, 6)}S")

    if show_plots:
        plt.plot(timestamps, vertical_acc)
        plt.scatter(steps_df['step_timestamp'], [0] * steps_df.shape[0])
        plt.scatter(default_steps_df['step_timestamp'], [0.1] * default_steps_df.shape[0])