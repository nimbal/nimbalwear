from datetime import timedelta
import math
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfilt, find_peaks, peak_widths, welch


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

#TODO: could probably just specify a low pass freq, init?
def lowpass_filter(acc_data, freq, order=2, cutoff_ratio=0.17):
    """
    Applies a lowpass filter on the accelerometer data
    """
    cutoff_freq = freq * cutoff_ratio
    sos = butter(N=order, Wn=cutoff_freq, btype='low', fs=freq, output='sos')
    acc_data = sosfilt(sos, acc_data)

    return acc_data, cutoff_freq


def flip_signal(acc_data, freq):
    """
    Finds orientation based on lowpassed signal and flips the signal
    """
    cutoff_freq = freq * 0.005
    sos = butter(N=1, Wn=cutoff_freq, btype='low', fs=freq, output='sos')
    orientation = sosfilt(sos, acc_data)
    flip_ind = np.where(orientation < -0.25)
    acc_data[flip_ind] = -acc_data[flip_ind]

    return acc_data


def state_space_steps(data, freq, start_time, loc=None, start_dp=0, end_dp=-1, pushoff_df=True):
    """
    Originally def step_detect(self)
    Detects the steps within the accelerometer data. Based on this paper:
    https://ris.utwente.nl/ws/portalfiles/portal/6643607/00064463.pdf
    ---
    Parameters
    ---
    data -> accelerometer data (acc_data, xz_data) from "get_acc_data_ssc"
    start_dp, end_dp -> indexed for start of step and end of step detection
    axis -> axis for vertical acceleration; default None but uses output from "get_acc_data_ssc"
    pushoff_df -> dataframe for pushoff detect; default is True to import premade pushoff df
    timestamps -> timestamps for data from "get_acc_data_ssc"
    ---
    Return
    ---
    steps_df -> dataframe with indexes of steps detected (beginning of step) from ssc algorithm
    """

    # define state space controller functions
    # detect_arr, state_arr, timestamps, step_indices, start, pushoff_time, foot_down_time
    def export_steps(detect_arr, state_arr, timestamps, step_indices, start_dp, pushoff_time, foot_down_time, loc,
                     success=True):
        """
        Export steps into a dataframe -  includes all potential push-offs and the state that they fail on
        """
        assert len(detect_arr) == len(timestamps)
        failed_step_indices = np.where(detect_arr > 0)[0]
        failed_step_timestamps = timestamps[failed_step_indices]

        error_mapping = {1: 'swing_down', 2: 'swing_up',
                         3: 'heel_strike_too_small', 4: 'too_close_to_next_i',
                         5: 'too_far_from_pushoff_mean', 6: 'mid_swing_peak_not_detected'}
        failed_step_state = list(map(error_mapping.get, detect_arr[failed_step_indices]))

        step_timestamps = timestamps[step_indices]

        swing_start = np.where((state_arr == 1) & (np.roll(state_arr, -1) == 2))[0]
        mid_swing = np.where((state_arr == 2) & (np.roll(state_arr, -1) == 3))[0]
        heel_strike = np.where((state_arr == 3) & (np.roll(state_arr, -1) == 4))[0]

        pushoff_start = swing_start - int(pushoff_time * freq)
        gait_cycle_end = heel_strike + int(foot_down_time * freq)
        step_durations = (gait_cycle_end - pushoff_start) / freq
        # avg_speed = [np.mean(xz_data[i:i + int(lengths * freq)]) * 9.81 * lengths for i, lengths in
        #            zip(step_indices, step_durations)]

        assert len(step_indices) == len(swing_start)
        assert len(step_indices) == len(mid_swing)
        assert len(step_indices) == len(heel_strike)

        successful_steps = pd.DataFrame({
            'step_timestamp': step_timestamps,
            'step_index': np.array(step_indices) + start_dp,
            'step_state': 'success',
            'swing_start_time': timestamps[swing_start],
            'mid_swing_time': timestamps[mid_swing],
            'heel_strike_time': timestamps[heel_strike],
            'swing_start_accel': data[swing_start],
            'mid_swing_accel': data[mid_swing],
            'heel_strike_accel': data[heel_strike],
            'step_duration': step_durations,
            'wear_loc': loc,
            'alg': 'ssc'
            # 'avg_speed': avg_speed
        })
        failed_steps = pd.DataFrame({
            'step_time': failed_step_timestamps,
            'step_index': np.array(failed_step_indices) + start_dp,
            'step_state': failed_step_state
        })
        if success == True:
            df = successful_steps
        else:
            df = pd.concat([successful_steps, failed_steps], sort=True)
            df = df.sort_values(by='step_index')
            df = df.reset_index(drop=True)

        return df

    def window_correlate(sig1, sig2):
        """
        Does cross-correlation between 2 signals over a window of indices
        """
        sig = np.array(max([sig1, sig2], key=len))
        window = np.array(min([sig1, sig2], key=len))

        engine = 'cython' if len(sig) < 100000 else 'numba'
        cc = pd.Series(sig
                       ).rolling(window=len(window)
                                 ).apply(lambda x: np.corrcoef(x, window)[0, 1], raw=True, engine=engine
                                         ).shift(-len(window) + 1
                                                 ).fillna(0
                                                          ).to_numpy()

        return cc

    def push_off_detection(data, pushoff_df, push_off_threshold, freq):
        """
        Detects the steps based on the pushoff_df, uses window correlate and cc threshold  to accept/reject pushoffs
        """
        pushoff_avg = pushoff_df['avg']

        cc_list = window_correlate(data, pushoff_avg)

        # TODO: Postponed -- DISTANCE CAN BE ADJUSTED FOR THE LENGTH OF ONE STEP RIGHT NOW ASSUMPTION IS THAT A PERSON CANT TAKE 2 STEPS WITHIN 0.5s
        pushoff_ind, _ = find_peaks(cc_list, height=push_off_threshold, distance=max(0.2 * freq, 1))

        return pushoff_ind

    def mid_swing_peak_detect(data, pushoff_ind, swing_phase_time, freq):
        """
        Detects a peak within the swing_detect window length - swing peak
        """
        swing_detect = int(freq * swing_phase_time)  # length to check for swing
        detect_window = data[pushoff_ind:pushoff_ind + swing_detect]
        peaks, prop = find_peaks(-detect_window,
                                 distance=max(swing_detect * 0.25, 1),
                                 prominence=0.2, wlen=swing_detect,
                                 width=[0 * freq, swing_phase_time * freq], rel_height=0.75)
        if len(peaks) == 0:
            return None

        results = peak_widths(-detect_window, peaks)
        prop['widths'] = results[0]

        return pushoff_ind + peaks[np.argmax(prop['widths'])]

    def swing_detect(data, pushoff_ind, mid_swing_ind):
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

        return (swing_down_cc, swing_up_cc)

    def heel_strike_detect(data, heel_strike_detect_time, window_ind, freq):
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

    # define thresholds
    push_off_threshold = 0.85
    swing_threshold = 0.5  # 0.5
    heel_strike_threshold = -5  # -5
    pushoff_time = 0.4  # 0.4 #tried 0.2 here
    swing_down_detect_time = 0.1  # 0.3
    swing_up_detect_time = 0.1  # 0.1
    swing_phase_time = swing_down_detect_time + swing_up_detect_time * 2
    heel_strike_detect_time = 0.5  # 0.8
    foot_down_time = 0.05  # 0.1 #tried 0.2 here

    label = 'StepDetector'
    pushoff_len = int(pushoff_time * freq)
    states = {1: 'stance', 2: 'push-off', 3: 'swing-up', 4: 'swing-down', 5: 'footdown'}
    state = states[1]

    # defining step pushoff thresholds
    if pushoff_df == True:  # importing static pushoff_df
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pushoff_df = pd.read_csv(os.path.join(dir_path, 'src', 'nimbalwear', 'data', 'pushoff_OND07_left.csv'))
    elif pushoff_df == False:
        print('No pushoff_df available, to fix define pushoff_df')

    if {'swing_down_mean', 'swing_down_std', 'swing_up_mean', 'swing_up_std', 'heel_strike_mean',
        'heel_strike_std'}.issubset(pushoff_df.columns):
        swing_phase_time = pushoff_df['swing_down_mean'].iloc[0] + pushoff_df['swing_down_std'].iloc[
            0] + pushoff_df['swing_up_mean'].iloc[0] + pushoff_df['swing_up_std'].iloc[0]
        swing_phase_time = max(swing_phase_time, 0.1)
        heel_strike_detect_time = 0.5 + pushoff_df['swing_up_mean'].iloc[0] + 2 * \
                                  pushoff_df['swing_up_std'].iloc[0]
        heel_strike_threshold = -3 - pushoff_df['heel_strike_mean'].iloc[0] / (
                2 * heel_strike_threshold)

    pushoff_ind = push_off_detection(data, pushoff_df, push_off_threshold, freq)
    end_pushoff_ind = pushoff_ind + pushoff_len
    state_arr = np.zeros(data.size)
    detects = {'push_offs': len(end_pushoff_ind), 'mid_swing_peak': [], 'swing_up': [], 'swing_down': [
    ], 'heel_strike': [], 'next_i': [], 'pushoff_mean': []}
    detect_arr = np.zeros(data.size)

    # initialize
    end_i = None
    step_indices = []
    step_lengths = []

    # run
    for count, i in tqdm(enumerate(end_pushoff_ind), total=len(end_pushoff_ind),
                         leave=False,
                         desc='%s: Step Detection' % label):
        # check if next index within the previous detection
        if end_i and i - pushoff_len < end_i:
            detects['next_i'].append(i - 1)
            continue

        # mean/std check for pushoff, state = 1
        pushoff_mean = np.mean(data[i - pushoff_len:i])
        upper = (pushoff_df['avg'] + pushoff_df['std'])
        lower = (pushoff_df['avg'] - pushoff_df['std'])
        if not np.any((pushoff_mean < upper) & (pushoff_mean > lower)):
            detects['pushoff_mean'].append(i - 1)
            continue

        mid_swing_i = mid_swing_peak_detect(data, i, swing_phase_time, freq)
        if mid_swing_i is None:
            detects['mid_swing_peak'].append(i - 1)
            continue

        accel_derivatives = heel_strike_detect(data, heel_strike_detect_time, mid_swing_i, freq)
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

    state_arr = state_arr
    step_indices = step_indices
    step_lengths = step_lengths
    detect_arr = detect_arr

    file_duration = len(data) / freq
    end_time = start_time + timedelta(0, file_duration)
    timestamps = np.asarray(pd.date_range(start=start_time, end=end_time, periods=len(data)))

    steps_df = export_steps(detect_arr, state_arr, timestamps, step_indices, start_dp, pushoff_time, foot_down_time,
                            loc)

    return steps_df


def fraccaro_gyro_steps(data, freq, start_time, loc=None, start_dp=0, end_dp=-1, steps_length=2, break_sec=2,
                        bout_steps=3):
    '''
    Detects the steps within the gyroscope data. Based on this paper:
    Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014). Real-world gyroscope-based gait event detection and gait feature extraction.
    '''

    # define functions
    def bw_filter(data, freq, fc, order):
        """
        Filter (filtfilt) data with dual pass lowpass butterworth filter
        """
        b, a = butter(N=order, Wn=fc, btype='low', output='ba', fs=freq)
        filtered_data = filtfilt(b, a, data)

        return filtered_data

    def find_adaptive_thresh(data, freq):
        '''
        Finds adaptive threshold on preprocessed data  with minimum 40 threshold

        B.R. Greene, et al., ”Adaptive estimation of temporal gait parameters using body-worn gyroscopes,”
        Proc. IEEE Eng. Med. Bio. Soc. (EMBC 2011), pp. 1296-1299, 2010 and outlined in Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014)
        '''
        data_2d = np.diff(data) / (1 / freq)

        thresh = np.mean(data[np.argpartition(data_2d, 10)[:10]]) * 0.2
        if thresh > 40:
            pass
        else:
            thresh = 40

        return thresh

    def remove_single_step_bouts(steps_df, steps_length):
        '''
        Step events are imported and bouts that have less than"steps_length" amount are removed from bouts_df
        '''
        sum_df = steps_df.groupby(['Bout_number']).count()
        sum_df.columns = ['Step_number', 'Step_index', 'Peak_times']

        sum_df.drop(sum_df[sum_df.Step_number < steps_length].index, inplace=True)
        bout_index = sum_df.index

        df = steps_df[steps_df.Bout_number.isin(bout_index)]
        df.reset_index(inplace=True, drop=True)

        return df

    def adjust_bout_number(steps_df):
        '''
        Renumbering the gait bouts for bouts_df after single step bouts are removed
        '''
        orig_bouts = steps_df.Bout_number
        num = 1
        for i in range(len(orig_bouts)):
            if i == 0:
                steps_df.loc[i, 2] = num
            else:
                if orig_bouts[i] > orig_bouts[i - 1]:
                    num = num + 1
                    steps_df.loc[i, 2] = num
                else:
                    steps_df.loc[i, 2] = num
        steps_df.drop('Bout_number', inplace=True, axis=1)
        steps_df.columns = ['Step', 'Step_index', 'Peak_times', 'Bout_number']

        return steps_df

    def get_bouts_info(steps_df):
        '''
        import steps_df and out bout_df
        '''
        bout_list = steps_df['Bout_number'].unique()
        bout_df = pd.DataFrame(columns=['Bout_number', 'Step_count', 'Start_time', 'End_time', 'Start_idx', 'End_idx'])
        for count, val in enumerate(bout_list):
            temp = steps_df[steps_df['Bout_number'] == bout_list[count]]
            step_count = len(temp)
            start_time = np.min(temp['Peak_times'])
            end_time = np.max(temp['Peak_times'])
            start_ind = np.min(temp['Step_index'])
            end_ind = np.max(temp['Step_index'])
            data = pd.DataFrame([[count + 1, step_count, start_time, end_time, start_ind, end_ind]],
                                columns=bout_df.columns)  # , "Cadence":cadence}
            bout_df = pd.concat([bout_df, data], ignore_index=True)

        return bout_df

    def find_steps_bouts_gyro(data, freq, timestamps, break_sec, steps_length, start, end):
        data = data[start:end]

        lf_data = bw_filter(data, freq, 3, 5)

        th1 = find_adaptive_thresh(lf_data, freq)

        idx_peaks, peak_hghts = find_peaks(x=data, height=th1,
                                           distance=40)  # at 50 samples/sec; 5 samples = 100 ms/0.1s; 10 samples = 200 ms/0.2s
        peak_heights = peak_hghts.get('peak_heights')

        peaks_diff = np.diff(idx_peaks)

        ge_break_ind = break_sec * freq
        bool_diff = peaks_diff > ge_break_ind
        ind_ge_diff = [i for i, x in enumerate(bool_diff) if x]
        bouts = np.zeros(len(idx_peaks))

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
        step_events_df = pd.DataFrame({'Step': step_count, 'Step_index': idx_peaks, 'Bout_number': bouts})

        # timestamps
        step_events_df['Step_timestamp'] = timestamps[step_events_df['Step_index']]
        gait_bouts_df = remove_single_step_bouts(step_events_df, steps_length)
        gait_bouts_df = adjust_bout_number(gait_bouts_df)
        gait_bouts_df = get_bouts_info(gait_bouts_df)
        step_events_df['Bout_number'] = 0
        for i in range(len(gait_bouts_df)):
            bool = (step_events_df.Step_index >= gait_bouts_df.Start_idx[i]) & (
                    step_events_df.Step_index <= gait_bouts_df.End_idx[i])
            idx = step_events_df.index[bool]
            step_events_df.Bout_number.iloc[idx] = gait_bouts_df.Bout_number[i]

        step_events_df.columns = ['step_number', 'step_index', 'bout_number', 'step_timestamp']
        bout_steps = step_events_df[step_events_df['bout_number'] != 0]
        bout_steps['foot'] = loc
        bout_steps['alg'] = 'fraccaro_gyro'

        return step_events_df, gait_bouts_df, bout_steps  # peak_heights

    file_duration = len(data) / freq
    end_time = start_time + timedelta(0, file_duration)
    timestamps = np.asarray(pd.date_range(start=start_time, end=end_time, periods=len(data)))

    _, _, bout_steps = find_steps_bouts_gyro(data, freq, timestamps, break_sec, steps_length, start_dp, end_dp)

    return bout_steps


def detect_steps(ra_data=None, la_data=None, data_type='accelerometer', data=None, left_right=None, loc='ankle',
                 start=0, end=-1, start_time=None, freq=None, orient_signal=True, low_pass=True):
    '''
    Parameters
    ---
    ra_data -> right side data; default None
    la_data -> left side data; default None
    data_type -> specify data to use to detect steps (what data type is in ra_data/la_data/data?); default accelerometer
    data -> data is for data input that is not ra_data or la_data; if this is defined then 'loc' needs to be defined
    left_right -> define wear side; 'left' , 'right', 'bilateral' or None (if trunk is true)
    loc -> define wear location; 'ankle', 'thigh','trunk'
    start -> where do you want to start your step detection; this can be a index value or datetime; default 0
    start_datetime -> if start is a datetime then this needs to be defined
    end -> where do you want step detection to end; this can be a index value or datetime; default 0

    freq -> sample frequency
    orient_signal -> check to see if polarity of signal is correct; default True
    low_pass -> low_pass filter the accelerometer data; default True
    ---
    Returns
    ---
    steps_df -> dataframe with detected steps
    '''

    #define functions

    #run steps_detect here
    if data_type == 'accelerometer':
        #ankle
        if (left_right == 'right') | (left_right=='bilateral') & (loc == 'ankle'):
                print('Finding steps: Right ankle, acceleration, state space controller.')

                ra_data = ra_data if ra_data is not None else data
                ra_data = ra_data if len(ra_data.shape)<2 else detect_vert(ra_data)

                if orient_signal:
                    ra_data = flip_signal(ra_data, freq)

                if low_pass:
                    ra_data, _ = lowpass_filter(ra_data, freq)

                right_steps_df = state_space_steps(ra_data, freq, start_time, loc='right', start_dp=start, end_dp=end, pushoff_df=True)

        if (left_right == 'left') | (left_right=='bilateral') & (loc == 'ankle'):
            print('Finding steps: Left ankle, acceleration, state space controller.')

            la_data = la_data if la_data is not None else data
            la_data = la_data if len(la_data.shape) < 2 else detect_vert(la_data)

            if orient_signal:
                la_data = flip_signal(la_data, freq)

            if low_pass:
                la_data, _ = lowpass_filter(la_data, freq)

            left_steps_df = state_space_steps(la_data, freq, start_time, loc='left', start_dp=start, end_dp=end,
                                              pushoff_df=True)

        if loc == 'trunk':
            print('Trunk step detection unavailable.')

    elif data_type == 'gyroscope':
        if (left_right == 'right') | (left_right== 'bilateral') & (loc == 'ankle'):
            print('Finding steps: Right ankle, gyroscope, Fraccaro algorithm.')

            ra_data = ra_data if ra_data is not None else data

            right_steps_df = fraccaro_gyro_steps(ra_data, freq, start_time, loc='right', start_dp=start, end_dp=end)

        if (left_right == 'left') | (left_right== 'bilateral') & (loc == 'ankle'):
            print('Finding steps: Left ankle, gyroscope, Fraccaro algorithm.')
            la_data = la_data if la_data is not None else data

            left_steps_df= fraccaro_gyro_steps(la_data, freq, start_time, loc='left', start_dp=start, end_dp=end)

        if (left_right == 'right') | (left_right == 'bilateral') & (loc == 'thigh'):
            print('Thigh step detetion unavailable')

        if (left_right == 'left') | (left_right == 'bilateral') & (loc == 'thigh'):
            print('Thigh step detetion unavailable')
    else:
        print('No data type defined')

    #create steps_df
    if left_right == 'bilateral':
        steps_df = pd.concat([right_steps_df, left_steps_df]).sort_values(by=['step_timestamp'])
        steps_df['step_number'] = np.arange(1,steps_df.shape[0]+1)
    elif left_right == 'right':
        steps_df = right_steps_df
    elif left_right == 'left':
        steps_df = left_steps_df
    else:
        print('No wear location specified. Specify loc as "right", "left", "bilateral". If loc is"trunk" or "thigh" analysis unavailable.')

    return steps_df

def get_walking_bouts(steps_df=None, min_bout_length=15, max_between_bouts=10, freq=None):
    """
 Parameters
    ---
    steps_df -> detect_steps output
    initiate_time -> amount of time (in seconds) steps need to be detected before bout is initiated
    mrp -> maximum resting period; amount of time (in seconds) with no steps before bout is terminated
    freq -> sampleing frequency
    stat_type -> input for bout_stats; how do you want the bouts group? 'daily' is the only operating type at the moment
    single leg -> does steps_df have one or two legs? should we double the step count to get total steps? True or False
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

########################################################################################################################
if __name__ == '__main__':

    from src.nimbalwear import Device

    ankle = Device()

    #GNAC testing
    subj = "OND06_1027_01"
    ankle_path = fr'W:\NiMBaLWEAR\OND06\processed\standard_device_edf\GNAC\{subj}_GNAC_LAnkle.edf'
    if os.path.exists(ankle_path):
        ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND06\processed\standard_device_edf\GNAC\{subj}_GNAC_LAnkle.edf')
    else:
        ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\sensor_edf\{subj}_GNAC_RAnkle.edf')

    ## get signal labels
    index_dict = {"accel_x": ankle.get_signal_index('Accelerometer x'),
                  "accel_y": ankle.get_signal_index('Accelerometer y'),
                  "accel_z": ankle.get_signal_index('Accelerometer z')}
    ##get signal frequencies needed for step detection
    fs = ankle.signal_headers[index_dict['accel_x']]['sample_rate']

    acc_data = np.array(ankle.signals[0:3])
    vertical_acc = np.array(ankle.signals[index_dict['accel_y']])

    data_start_time = ankle.header["start_datetime"]  # if start is None else start

    # #Input for detect steps is "Device" obj
    #def detect_steps(ra_data=None, la_data=None, data_type='accelerometer', loc=None, data=None, start=0, end=-1, freq=None, orient_signal=True, low_pass=True):
    steps_df = detect_steps(ra_data=None, la_data=vertical_acc, data_type='accelerometer', left_right='left', loc='ankle', data=None, start_time=data_start_time, start=100000, end=200000, freq=fs)

###############################################################################
   #  #AXV6 testing
   #  subj = "OND09_0011_01"
   #  ankle_path = fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_LAnkle.edf'
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
    #---
    #
    # get walking bouts should run on any detect_steps output (steps_df
    bouts = get_walking_bouts(steps_df=steps_df, initiate_time=15, mrp=10, freq=fs)
    bout_stats = gait_stats(bouts, stat_type='daily', single_leg=False)
