import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks


def bw_filter(data, freq, fc, order):
    """
    Filter (sosfiltfilt) data with dual pass lowpass butterworth filter
    """
    sos = butter(N=order, Wn=fc, btype='low', output='sos', fs=freq)
    filtered_data = sosfiltfilt(sos, data)

    return filtered_data


def find_adaptive_thresh(data, freq):
    """
    Finds adaptive threshold on preprocessed data  with minimum 40 threshold

    B.R. Greene, et al., ”Adaptive estimation of temporal gait parameters using body-worn gyroscopes,”
    Proc. IEEE Eng. Med. Bio. Soc. (EMBC 2011), pp. 1296-1299, 2010 and outlined in Fraccaro, P., Coyle, L., Doyle, J.,
    & O'Sullivan, D. (2014)
    """
    data_2d = np.diff(data) / (1 / freq)

    thresh = np.mean(data[np.argpartition(data_2d, 10)[:10]]) * 0.2
    if thresh > 40:
        pass
    else:
        thresh = 40

    return thresh


def adjust_bout_number(steps):
    """
    Renumbering the gait bouts for bouts_df after single step bouts are removed
    """

    old = steps['bout_num'].unique().tolist()
    old.sort()
    new = range(len(old))
    bout_dict = dict(zip(old,new))

    steps['bout_num'] = [bout_dict[b] for b in steps['bout_num']]

    return steps


def remove_short_bouts(steps, steps_length):
    """
    Step events are imported and bouts that have less than"steps_length" amount are removed from bouts_df
    """

    sum_df = steps.groupby(['bout_num']).count()
    sum_df.columns = ['step_num', 'step_id']

    sum_df.drop(sum_df[sum_df['step_num'] < steps_length].index, inplace=True)
    bout_index = sum_df.index

    steps['bout_num'][~steps['bout_num'].isin(bout_index)] = 0

    steps = adjust_bout_number(steps)

    return steps


def get_bouts_info(steps):
    """
    import steps_df and out bout_df
    """

    bout_list = steps['bout_num'].unique()
    bout_list = np.delete(bout_list, np.where(bout_list == 0)).tolist()

    step_count = []
    start_idx = []
    end_idx = []

    for bout_num in bout_list:
        bout_steps = steps[steps['bout_num'] == bout_num]
        step_count.append(len(bout_steps))
        start_idx.append(min(bout_steps['step_idx']))
        end_idx.append(max(bout_steps['step_idx']))

    bouts = pd.DataFrame({'bout_num': bout_list, 'start_idx': start_idx, 'end_idx': end_idx, 'step_count': step_count})

    return bouts


def detect_steps(gyro, freq):

    lf_data = bw_filter(gyro, freq, 3, 5)

    th1 = find_adaptive_thresh(lf_data, freq)

    idx_peaks, peak_hghts = find_peaks(x=gyro, height=th1, distance=(.8 * freq))

    step_count = range(1, len(idx_peaks) + 1)
    steps = pd.DataFrame({'step_num': step_count, 'step_idx': idx_peaks})

    return steps


def define_bouts(steps, freq, max_break=2, min_steps=2, remove_unbouted=True):

    idx_peaks = steps['step_idx'].to_numpy()

    peaks_diff = np.diff(idx_peaks)

    ge_break_ind = max_break * freq
    bool_diff = peaks_diff > ge_break_ind
    ind_ge_diff = [i for i, x in enumerate(bool_diff) if x]
    bouts = np.zeros(len(idx_peaks), dtype=int)

    for count, x in enumerate(ind_ge_diff):
        if count < len(ind_ge_diff) - 1:
            if x == 0:
                bouts[count] = 1
                continue
            elif ind_ge_diff[count] - ind_ge_diff[count + 1] == 1:
                bouts[count] = count + 1
                continue
            else:
                if count == 0:
                    bouts[:(ind_ge_diff[count + 1])] = count + 1
                else:
                    bouts[(ind_ge_diff[count - 1] + 1):(ind_ge_diff[count + 1] + 1)] = count + 1
        elif count == len(ind_ge_diff) - 1:
            bouts[(ind_ge_diff[count - 1] + 1):ind_ge_diff[count] + 1] = count + 1
            bouts[ind_ge_diff[count] + 1:] = count + 2

    step_count = range(1, len(idx_peaks) + 1)
    new_steps = pd.DataFrame({'step_num': step_count, 'step_idx': idx_peaks, 'bout_num': bouts.tolist()})

    new_steps = remove_short_bouts(new_steps, min_steps)

    gait_bouts = get_bouts_info(new_steps)

    if remove_unbouted:
        new_steps = new_steps[new_steps['bout_num'] != 0]

    return new_steps[['step_num', 'bout_num', 'step_idx']], gait_bouts


def fraccaro_gyro_steps(gyro, freq, start_time=None, min_steps=2, max_break=2, remove_unbouted=True):
    """
    Detects the steps within the gyroscope data. Based on this paper:
    Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014). Real-world gyroscope-based gait event detection and
    gait feature extraction.
    """

    # detect_steps
    steps_df = detect_steps(gyro, freq)

    # get bouted steps
    steps_df, bouts_df = define_bouts(steps_df, freq, max_break=max_break, min_steps=min_steps,
                                      remove_unbouted=remove_unbouted)

    if start_time is not None:

        timestamps = pd.date_range(start=start_time, periods=len(gyro), freq=f"{1/freq}S")

        steps_df.insert(loc=2, column='step_time', value=timestamps[steps_df['step_idx']])

        bouts_df.insert(loc=1, column='start_time', value=timestamps[bouts_df['start_idx']])
        bouts_df.insert(loc=2, column='end_time', value=timestamps[bouts_df['end_idx']])

    return steps_df, bouts_df


if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    from src.nimbalwear import Device

    show_plots = True

    # AXV6 testing
    ankle_path = Path("W:/prd/nimbalwear/OND09/wearables/device_edf_cropped/OND09_SBH0011_01_AXV6_RAnkle.edf")
    ankle = Device()
    ankle.import_edf(ankle_path)

    ## get signal idxs
    gyro_z_idx = ankle.get_signal_index('Gyroscope z')

    ##get signal frequencies needed for step detection
    fs = ankle.signal_headers[gyro_z_idx]['sample_rate']

    gyro_z = ankle.signals[gyro_z_idx]

    data_start_time = ankle.header['start_datetime']  # if start is None else start

    gyro_steps_df, gyro_bouts_df = fraccaro_gyro_steps(gyro=gyro_z, freq=fs, start_time=data_start_time, min_steps=3,
                                                       max_break=2, remove_unbouted=False)

    # steps_df = detect_steps(ra_data=sag_gyro, la_data=sag_gyro, data_type='gyroscope', left_right='bilateral', loc='ankle', data=None, start_time=data_start_time, start=0, end=-1, freq=fs)
    #
    # ---
    #
    # get walking bouts should run on any detect_steps output (steps_df
    # bouts = get_walking_bouts(steps_df=steps_df, min_bout_length=15, max_between_bouts=10, freq=fs)
    # bout_stats = gait_stats(bouts, stat_type='daily', single_leg=True)
