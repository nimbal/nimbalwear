from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .gait_accel import state_space_accel_steps
from .gait_gyro import fraccaro_gyro_steps


def lowpass_filter(data, fs, cutoff_freq, order=2):
    """
    Applies a lowpass filter on the accelerometer data
    """

    sos = butter(N=order, Wn=cutoff_freq, btype='low', fs=fs, output='sos')
    filt_data = sosfiltfilt(sos, data)

    return filt_data


def flip_signal(data, freq):
    """
    Finds orientation based on lowpassed signal and flips the signal
    """
    cutoff_freq = freq * 0.005
    sos = butter(N=1, Wn=cutoff_freq, btype='low', fs=freq, output='sos')
    orientation = sosfiltfilt(sos, data)
    flip_ind = np.where(orientation < -0.25)
    data[flip_ind] = -data[flip_ind]

    return data


def detect_vert(axes, method='adg'):
    """
    NOTE: To improve function when passing in axes:
                - remove axes that are unlikely to be the vertical axis
                - remove data points that are known to be nonwear or lying down
                - OR only pass data from a known bout of standing

    Parameters
    ---
    axes -> all axs of accelerometer sensors
    method-> adg , mam
    """
    axes_arr = np.array(axes)

    vert_idx = None

    if method == 'mam':

        test_stats = np.mean(np.abs(axes_arr), axis=1)
        vert_idx = np.argmax(test_stats)

    elif method == 'adg':

        test_stats = np.abs(1 - np.abs(np.mean(axes_arr, axis=1)))
        vert_idx = np.argmin(test_stats)

    vert_data = axes[vert_idx]

    return vert_data  # , vert_idx, test_stats


def detect_steps(right_data=None, left_data=None, mid_data=None, loc='ankle', data_type='accel',
                 start_time=None, freq=None, orient_signal=True, low_pass=12):
    """
    Parameters
    ---
    right_data -> right side data; default None
    left_data -> left side data; default None
    mid_data - > midline/trunk data; default None
    loc -> define wear location; 'ankle', 'thigh','trunk'
    data_type -> specify data to use to detect steps (what data type is in ra_data/la_data/data?); default accelerometer
    data -> data is for data input that is not ra_data or la_data; if this is defined then 'loc' needs to be defined
    left_right -> define wear side; 'left' , 'right', 'bilateral' or None (if trunk is true)

    start -> where do you want to start your step detection; this can be a index value or datetime; default 0
    start_datetime -> if start is a datetime then this needs to be defined
    end -> where do you want step detection to end; this can be a index value or datetime; default 0

    freq -> sample frequency
    orient_signal -> check to see if polarity of signal is correct; default True
    low_pass -> cut off of low_pass filter on the accelerometer data or None if no filter; default 12
    ---
    Returns
    ---
    steps_df -> dataframe with detected steps
    """

    data_type = 'accel' if data_type in ['a', 'acc', 'accel', 'accelerometer'] else data_type
    data_type = 'gyro' if data_type in ['g', 'gyr', 'gyro', 'gyroscope'] else data_type

    left_steps = pd.DataFrame()
    right_steps = pd.DataFrame()

    # TODO: add parameters to settings

    if data_type == 'accel':
        if loc == 'ankle':
            if right_data is not None:
                print('Finding steps: Right ankle, acceleration, state space controller.')

                # right_data = right_data if len(right_data.shape) < 2 else detect_vert(right_data)

                if orient_signal:
                    right_data = flip_signal(right_data, freq)
                if low_pass is not None:
                    right_data = lowpass_filter(right_data, freq, low_pass)

                right_steps = state_space_accel_steps(vert_accel=right_data, freq=freq, start_time=start_time,
                                                      pushoff_threshold=0.85, pushoff_time=0.4,
                                                      swing_down_detect_time=0.1, swing_up_detect_time=0.1,
                                                      heel_strike_detect_time=0.5, heel_strike_threshold=-5,
                                                      foot_down_time=0.05, success=True, update_pars=True,
                                                      return_default=False)

                right_steps['loc'] = loc
                right_steps['side'] = 'right'
                right_steps['data_type'] = data_type
                right_steps['alg'] = 'ssc'

            if left_data is not None:
                print('Finding steps: Left ankle, acceleration, state space controller.')

                # left_data = left_data if len(left_data.shape) < 2 else detect_vert(left_data)

                if orient_signal:
                    left_data = flip_signal(left_data, freq)

                if low_pass is not None:
                    left_data = lowpass_filter(left_data, freq, low_pass)

                left_steps = state_space_accel_steps(vert_accel=left_data, freq=freq, start_time=start_time,
                                                     pushoff_threshold=0.85, pushoff_time=0.4,
                                                     swing_down_detect_time=0.1, swing_up_detect_time=0.1,
                                                     heel_strike_detect_time=0.5, heel_strike_threshold=-5,
                                                     foot_down_time=0.05, success=True, update_pars=True,
                                                     return_default=False)

                left_steps['loc'] = loc
                left_steps['side'] = 'left'
                left_steps['data_type'] = data_type
                left_steps['alg'] = 'ssc'

        elif loc == 'trunk':
            print('Trunk step detection unavailable.')
        else:
            print(f'Invalid loc: {loc}')

    elif data_type == 'gyro':
        if loc == 'ankle':

            if right_data is not None:
                print('Finding steps: Right ankle, gyroscope, Fraccaro algorithm.')

                right_steps = fraccaro_gyro_steps(gyro=right_data, freq=freq, start_time=start_time)

                right_steps['loc'] = loc
                right_steps['side'] = 'right'
                right_steps['data_type'] = data_type
                right_steps['alg'] = 'fraccaro'

            if left_data is not None:
                print('Finding steps: Left ankle, gyroscope, Fraccaro algorithm.')

                left_steps = fraccaro_gyro_steps(gyro=left_data, freq=freq, start_time=start_time)

                left_steps['loc'] = loc
                left_steps['side'] = 'left'
                left_steps['data_type'] = data_type
                left_steps['alg'] = 'fraccaro'

        elif loc == 'thigh':
            print('Thigh step detection unavailable')
        else:
            print(f'Invalid loc: {loc}')

    else:
        print(f'Invalid data type: {data_type}')

    steps = pd.DataFrame()

    # create steps_df
    if (not left_steps.empty) & (not right_steps.empty):
        steps = pd.concat([right_steps, left_steps]).sort_values(by=['step_time'])
        steps['step_num'] = np.arange(1, steps.shape[0]+1)
    elif not right_steps.empty:
        steps = right_steps
    elif not left_steps.empty:
        steps = left_steps
    else:
        print('Unable to perform step detection')

    return steps


def adjust_bout_number(steps):
    """
    Renumbering the gait bouts for bouts_df after single step bouts are removed
    """

    old = steps['gait_bout_num'].unique().tolist()
    old.sort()
    new = range(len(old))
    bout_dict = dict(zip(old, new))

    steps['gait_bout_num'] = [bout_dict[b] for b in steps['gait_bout_num']]

    return steps


def remove_short_bouts(steps, steps_length):
    """
    Step events are imported and bouts that have less than"steps_length" amount are removed from bouts_df
    """

    sum_df = steps.groupby(['gait_bout_num']).count()
    # sum_df.columns = ['step_num', 'step_id']

    sum_df.drop(sum_df[sum_df['step_num'] < steps_length].index, inplace=True)
    bout_index = sum_df.index

    new_steps = steps.copy()
    new_steps['gait_bout_num'][~new_steps['gait_bout_num'].isin(bout_index)] = 0

    new_steps = adjust_bout_number(new_steps)

    return new_steps


def get_bouts_info(steps):
    """
    import steps_df and out bout_df
    """

    bout_list = steps['gait_bout_num'].unique()
    bout_list = np.delete(bout_list, np.where(bout_list == 0)).tolist()

    step_count = []
    start_idx = []
    end_idx = []

    for bout_num in bout_list:
        bout_steps = steps[steps['gait_bout_num'] == bout_num]
        step_count.append(len(bout_steps))
        start_idx.append(min(bout_steps['step_idx']))
        end_idx.append(max(bout_steps['step_idx']))

    bouts = pd.DataFrame({'gait_bout_num': bout_list, 'start_idx': start_idx, 'end_idx': end_idx, 'step_count': step_count})

    return bouts


def define_bouts(steps, freq, start_time=None, max_break=2, min_steps=2, remove_unbouted=True):

    idx_peaks = steps['step_idx'].to_numpy()

    peaks_diff = np.diff(idx_peaks)

    ge_break_ind = max_break * freq
    bool_diff = peaks_diff > ge_break_ind
    ind_ge_diff = [i for i, x in enumerate(bool_diff) if x]
    bout_num = np.zeros(len(idx_peaks), dtype=int)

    for count, x in enumerate(ind_ge_diff):
        if count < len(ind_ge_diff) - 1:
            if x == 0:
                bout_num[count] = 1
            elif ind_ge_diff[count] - ind_ge_diff[count + 1] == 1:
                bout_num[count] = count + 1
            else:
                if count == 0:
                    bout_num[:(ind_ge_diff[count + 1])] = count + 1
                else:
                    bout_num[(ind_ge_diff[count - 1] + 1):(ind_ge_diff[count + 1] + 1)] = count + 1
        elif count == len(ind_ge_diff) - 1:
            bout_num[(ind_ge_diff[count - 1] + 1):ind_ge_diff[count] + 1] = count + 1
            bout_num[ind_ge_diff[count] + 1:] = count + 2

    new_steps = steps.copy()
    new_steps.insert(loc=1, column="gait_bout_num", value=bout_num.tolist())

    new_steps = remove_short_bouts(new_steps, min_steps)

    bouts = get_bouts_info(new_steps)

    if remove_unbouted:
        new_steps = new_steps[new_steps['gait_bout_num'] != 0]

    if start_time is not None:

        bout_start_times = pd.Series([start_time + timedelta(seconds=(i / freq)) for i in bouts['start_idx']])
        bout_end_times = pd.Series([start_time + timedelta(seconds=(i / freq)) for i in bouts['end_idx']])

        bouts.insert(loc=1, column='start_time', value=bout_start_times)
        bouts.insert(loc=2, column='end_time', value=bout_end_times)

    return new_steps, bouts


def gait_stats(bouts, stat_type='daily', single_leg=False):

    bouts = bouts.copy()

    bouts['date'] = pd.to_datetime(bouts['start_time']).dt.date
    bouts['duration'] = [round((x['end_time'] - x['start_time']).total_seconds()) for i, x in bouts.iterrows()]

    # if only steps from one leg then double step counts
    bouts['step_count'] = bouts['step_count'] * 2 if single_leg else bouts['step_count']

    if stat_type == 'daily':

        stats = pd.DataFrame(columns=['day_num', 'date', 'type', 'longest_bout_time', 'longest_bout_steps',
                                      'bouts_over_3min', 'total_steps'])

        day_num = 1

        for date, group_df in bouts.groupby('date'):
            day_stats = pd.DataFrame([[day_num, date, stat_type, group_df['duration'].max(),
                                       round(group_df['step_count'].max()),
                                       group_df.loc[group_df['duration'] > 180].shape[0],
                                       round(group_df['step_count'].sum())]], columns=stats.columns)

            stats = pd.concat([stats, day_stats], ignore_index=True)

            day_num += 1

    else:
        stats = pd.DataFrame()
        print('Invalid type selected.')

    return stats
