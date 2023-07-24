from datetime import timedelta
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, peak_widths

from .gait_accel import state_space_accel_steps
from .gait_gyro import fraccaro_gyro_steps

def lowpass_filter(acc_data, fs, cutoff_freq, order=2):
    """
    Applies a lowpass filter on the accelerometer data
    """

    sos = butter(N=order, Wn=cutoff_freq, btype='low', fs=fs, output='sos')
    acc_data = sosfiltfilt(sos, acc_data)

    return acc_data


def flip_signal(acc_data, freq):
    """
    Finds orientation based on lowpassed signal and flips the signal
    """
    cutoff_freq = freq * 0.005
    sos = butter(N=1, Wn=cutoff_freq, btype='low', fs=freq, output='sos')
    orientation = sosfiltfilt(sos, acc_data)
    flip_ind = np.where(orientation < -0.25)
    acc_data[flip_ind] = -acc_data[flip_ind]

    return acc_data


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

    test_stats = None
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
                 start=0, end=-1, start_time=None, freq=None, orient_signal=True, low_pass=12):
    '''
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
    '''

    #define functions
    data_type = 'a' if data_type in ['a', 'acc', 'accel', 'accelerometer'] else data_type
    data_type = 'g' if data_type in ['g', 'gyr', 'gyro', 'gyroscope'] else data_type

    left_steps_df = pd.DataFrame()
    right_steps_df = pd.DataFrame()

    # TODO: add parameters to settings

    #run steps_detect here
    if data_type == 'a':
        if loc == 'ankle': #ankle
            if right_data is not None:
                print('Finding steps: Right ankle, acceleration, state space controller.')

                # right_data = right_data if len(right_data.shape) < 2 else detect_vert(right_data)

                if orient_signal:
                    right_data = flip_signal(right_data, freq)
                if low_pass is not None:
                    right_data, _ = lowpass_filter(right_data, low_pass)

                right_steps_df = state_space_steps(right_data, freq, start_time, loc='right')

            if left_data is not None:
                print('Finding steps: Left ankle, acceleration, state space controller.')

                # left_data = left_data if len(left_data.shape) < 2 else detect_vert(left_data)

                if orient_signal:
                    left_data = flip_signal(left_data, freq)

                if low_pass is not None:
                    left_data, _ = lowpass_filter(left_data, freq, low_pass)

                left_steps_df = state_space_steps(left_data, freq, start_time, loc='left')

        elif loc == 'trunk':
            print('Trunk step detection unavailable.')
        else:
            print(f'Invalid loc: {loc}')

    elif data_type == 'g':
        if loc == 'ankle':

            if right_data is not None:
                print('Finding steps: Right ankle, gyroscope, Fraccaro algorithm.')

                right_steps_df = fraccaro_gyro_steps(right_data, freq, start_time, loc='right', start_dp=start, end_dp=end)

            if left_data is not None:
                print('Finding steps: Left ankle, gyroscope, Fraccaro algorithm.')

                left_steps_df = fraccaro_gyro_steps(left_data, freq, start_time, loc='left', start_dp=start, end_dp=end)

        elif loc == 'thigh':
            print('Thigh step detection unavailable')
        else:
            print(f'Invalid loc: {loc}')

    else:
        print(f'Invalid data type: {data_type}')

    #create steps_df
    if (not left_steps_df.empty) & (not right_steps_df.empty):
        steps_df = pd.concat([right_steps_df, left_steps_df]).sort_values(by=['step_timestamp'])
        steps_df['step_number'] = np.arange(1,steps_df.shape[0]+1)
    elif not right_steps_df.empty:
        steps_df = right_steps_df
    elif not left_steps_df.empty:
        steps_df = left_steps_df
    else:
        print('Unable to perform step detection')

    return steps_df


def get_bouts(steps_df, min_bout_length=15, max_between_bouts=10, freq=None):
    """
 Parameters
    ---
    steps_df -> detect_steps output
    min_bout_length -> amount of time (in seconds) steps need to be detected before bout is initiated
    max_between_bouts -> maximum resting period; amount of time (in seconds) with no steps before bout is terminated
    freq -> sampleing frequency
    ---
    Returns
    ---
    bouts -> dataframe with detected steps organized into bouts
    bout_stats -> summary of bout distribution for stats
    """

    steps = steps_df['step_index']  # step_detector.step_indices
    timestamps = steps_df['step_timestamp']  # step_detector.timestamps[steps]
    step_durations = steps_df['step_duration'] if 'step_durations' in steps_df.columns else None  # step_detector.step_lengths

    freq=int(freq)

    steps_df = pd.DataFrame({'step_index': steps, 'timestamp': timestamps, 'step_duration': step_durations})
    steps_df = steps_df.sort_values(by=['step_index'], ignore_index=True)

    # assumes Hz are the same
    bout_dict = {'start': [], 'end': [], 'step_count': [], 'start_time': [], 'end_time': []}
    if steps_df.empty:
        return pd.DataFrame(bout_dict)
    start_step = steps_df.iloc[0]  # start of bout step
    curr_step = steps_df.iloc[0]
    step_count = 1
    next_steps = None

    while curr_step is not None:

        # Assumes steps are not empty and finds the next step after the current step
        termination_bout_window = min_bout_length if next_steps is None else max_between_bouts
        termination_bout_window = pd.Timedelta(termination_bout_window, unit='sec')
        next_steps = steps_df.loc[(steps_df['timestamp'] <= termination_bout_window + curr_step['timestamp'])
                                  & (steps_df['timestamp'] > curr_step['timestamp'])]

        if not next_steps.empty:
            curr_step = next_steps.iloc[0]
            step_count += 1
        else:

            if step_count >= 3:
                start_ind = start_step['step_index']
                end_ind = curr_step['step_index'] + curr_step['step_duration'] if 'step_durations' in curr_step.index else curr_step['step_index']
                bout_dict['start'].append(start_ind)
                bout_dict['end'].append(end_ind)
                bout_dict['step_count'].append(step_count)
                bout_dict['start_time'].append(start_step['timestamp'])
                bout_dict['end_time'].append(
                    curr_step['timestamp'] + pd.Timedelta(curr_step['step_duration'] / freq, unit='sec')) if 'step_durations' in curr_step.index else bout_dict['end_time'].append(curr_step['timestamp'])

            # resets state and creates new bout
            step_count = 1
            next_curr_steps = steps_df.loc[steps_df['timestamp'] > curr_step['timestamp']]
            curr_step = next_curr_steps.iloc[0] if not next_curr_steps.empty else None
            start_step = curr_step
            next_steps = None

    bouts = pd.DataFrame(bout_dict)

    return bouts


def gait_stats(bouts, stat_type='daily', single_leg=False):

    bouts['date'] = pd.to_datetime(bouts['start_time']).dt.date
    bouts['duration'] = [round((x['end_time'] - x['start_time']).total_seconds()) for i, x in bouts.iterrows()]

    # if only steps from one leg then double step counts
    bouts['step_count'] = bouts['step_count'] * 2 if single_leg else bouts['step_count']

    if stat_type == 'daily': #hour #week #waking hours #sleep hours

        gait_stats = pd.DataFrame(columns=['day_num', 'date', 'type', 'longest_bout_time', 'longest_bout_steps',
                                           'bouts_over_3min', 'total_steps'])

        day_num = 1

        for date, group_df in bouts.groupby('date'):
            day_gait_stats = pd.DataFrame([[day_num, date, stat_type, group_df['duration'].max(),
                                            round(group_df['step_count'].max()),
                                            group_df.loc[group_df['duration'] > 180].shape[0],
                                            round(group_df['step_count'].sum())]], columns=gait_stats.columns)

            gait_stats = pd.concat([gait_stats, day_gait_stats], ignore_index=True)

            day_num += 1

    else:
        gait_stats = pd.DataFrame()
        print('Invalid type selected.')

    return gait_stats


if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    from . import Device

    # GNAC testing
    ankle_path = Path("W:/NiMBaLWEAR/OND06/processed/standard_device_edf/GNAC/OND06_1027_01_GNAC_LAnkle.edf")
    ankle = Device()
    ankle.import_edf(ankle_path)

    ## get signal idxs
    y_idx = ankle.get_signal_index('Accelerometer y')

    ##get signal frequencies needed for step detection
    fs = ankle.signal_headers[y_idx]['sample_rate']

    vertical_acc = ankle.signals[y_idx]

    data_start_time = ankle.header['start_datetime']  # if start is None else start

    dir_path = os.path.dirname(os.path.realpath(__file__))
    pushoff_df = pd.read_csv(os.path.join(dir_path, 'data', 'pushoff_df.csv'))

    # state_arr, detect_arr, step_indices, step_lengths = detect_steps(vert_accel=vertical_acc, freq=fs,
    #                                                                  pushoff_df=pushoff_df, start_time=data_start_time)

    steps_df, default_steps_df = state_space_steps(vert_accel=vertical_acc, freq=fs, start_time=data_start_time,
                                                   update_pars=True, return_default=True)

    file_duration = len(vertical_acc) / fs
    end_time = data_start_time + timedelta(0, file_duration)
    timestamps = np.asarray(pd.date_range(start=data_start_time, end=end_time, periods=len(vertical_acc)))

    plt.plot(timestamps, vertical_acc)
    plt.scatter(steps_df['step_timestamp'], [0] * steps_df.shape[0])
    plt.scatter(default_steps_df['step_timestamp'], [0.1] * default_steps_df.shape[0])


    ###############################################################################
    #  #AXV6 testing
    #  subj = "OND09_0011_01"
    #  ankle_path = 'W:/NiMBaLWEAR/OND09/wearables/device_edf_cropped/OND09_0011_01_AXV6_RAnkle.edf'
    #  if os.path.exists(ankle_path):
    #       ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_LAnkle.edf')
    #  else:
    #       ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_RAnkle.edf')
    #
    # # get signal labels
    #  index_dict = {"accel_x": ankle.get_signal_index('Accelerometer x'),
    #                "accel_y": ankle.get_signal_index('Accelerometer y'),
    #                "accel_z": ankle.get_signal_index('Accelerometer z'),
    #                "gyro_x": ankle.get_signal_index('Gyroscope x'),
    #                "gyro_y": ankle.get_signal_index('Gyroscope y'),
    #                "gyro_z": ankle.get_signal_index('Gyroscope z')}
    #  ##get signal frequencies needed for step detection
    #  fs = ankle.signal_headers[index_dict['gyro_z']]['sample_rate']
    #
    #  gyro_data = np.array([ankle.signals[index_dict['gyro_x']], ankle.signals[index_dict['gyro_y']], ankle.signals[index_dict['gyro_z']]])
    #  acc_data = np.array([ankle.signals[index_dict['accel_x']], ankle.signals[index_dict['accel_y']], ankle.signals[index_dict['accel_z']]])
    #  sag_gyro = np.array(ankle.signals[index_dict['gyro_z']])
    #
    #  data_start_time = ankle.header["start_datetime"]  # if start is None else start

    # steps_df = detect_steps(ra_data=sag_gyro, la_data=sag_gyro, data_type='gyroscope', left_right='bilateral', loc='ankle', data=None, start_time=data_start_time, start=0, end=-1, freq=fs)
    #
    # ---
    #
    # get walking bouts should run on any detect_steps output (steps_df
    # bouts = get_walking_bouts(steps_df=steps_df, min_bout_length=15, max_between_bouts=10, freq=fs)
    # bout_stats = gait_stats(bouts, stat_type='daily', single_leg=True)
