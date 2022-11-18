from datetime import timedelta
import math
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfilt, find_peaks, peak_widths, welch
import matplotlib.pyplot as plt
import nimbalwear



#AccelReader declassed

def get_acc_data(accelerometer=None, axis=None, orient_signal=True, low_pass=True):
    """
        Parameters
        ---
        obj -> accelerometer object as read by nimbalwear.Device()
        axis  -> can specific the vertical axis; default is 'None' determines vertical
        orient_signal -> flips the vertical axis if needed
        low_pass -> low pass filter to remove noise
        ---
        Returns
        ---
        frequency, accelerometer data, xz_data, timestamps, axis
        """

    def detect_vert(axes, method='adg'):
        """
        NOTE: To improve function when passing in axes:
                    - remove axes that are unlikely to be the vertical axis
                    - remove data points that are known to be nonwear or lying down
                    - OR only pass data from a known bout of standing

        Parameters
        ---
        axes -> all axs of accelerometer sensors
        method-> no clue tbh;
            adg = something with absolute _?_  gravity??
            mam = something with accelerometer magnitude?

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

        return vert_idx, test_stats


    def lowpass_filter(acc_data, freq, order=2, cutoff_ratio=0.17):
        """
        Applies a lowpass filter on the accelerometer data

        Parameters
        ---
        acc_data
        freq
        order -> filter order; default is 2nd order
        cutoff_ratio -> used to determine the cutoff frequency

        Ouput
        ---
        acc_data low pass filtered, cutoff_freq
        """
        cutoff_freq = freq * cutoff_ratio
        sos = butter(N=order, Wn=cutoff_freq,
                     btype='low', fs=freq, output='sos')
        acc_data = sosfilt(sos, acc_data)

        return acc_data, cutoff_freq


    def flip_signal(acc_data, freq):
        """
        Finds orientation based on lowpassed signal and flips the signal

        Parameters
        ---
        acc_data -> accelerometer np.array
        freq -> sampling frequency stored in header

        Output
        ---
        acc_data
        """

        cutoff_freq = freq * 0.005
        sos = butter(N=1, Wn=cutoff_freq, btype='low', fs=freq, output='sos')
        orientation = sosfilt(sos, acc_data)
        flip_ind = np.where(orientation < -0.25)
        acc_data[flip_ind] = -acc_data[flip_ind]

        return acc_data

    all_data = np.array(accelerometer.signals)
    accel_axes = [0, 1, 2]
    if axis is not None:
        accel_axes.remove(axis)
        acc_data = all_data[axis]
        xz_data = np.sqrt(all_data[accel_axes[0]] ** 2 + all_data[accel_axes[1]] ** 2)
    else:
        axis_index, test_stats = detect_vert(all_data[0:2])  # assumes vertical is 0 or 1
        other_axes = np.delete(np.arange(all_data.shape[0]), axis_index)
        axis = accel_axes[axis_index]
        acc_data = all_data[axis_index]
        xz_data = np.sqrt((all_data[other_axes] ** 2).sum(axis=0))

    freq = accelerometer.signal_headers[axis]['sample_rate']
    start_time = accelerometer.header['start_datetime']
    file_duration = len(acc_data) / freq
    end_time = start_time + timedelta(0, file_duration)
    timestamps = np.asarray(pd.date_range(start=start_time, end=end_time, periods=len(acc_data)))

    if orient_signal:
        acc_data = flip_signal(acc_data, freq)

    if low_pass:
        acc_data = lowpass_filter(acc_data, freq)

    return freq, acc_data, xz_data, timestamps, axis


# Stepdetector declassed
def detect_steps(device=None, data=None, start=0, end=-1,  freq=None, axis=None,  timestamps=None, xz_data=None):
    '''
    Parameters
    ---
    device -> input nw.Device() object; default is None
    ---
    Returns
    ---
    steps_df -> dataframe with detected steps
    '''

    #acc_step_detect_ssc(data=None, start_dp=1, end_dp=-1, pushoff_df=None)
    def detect_steps_ssc(device = None, data=None,  start_dp=start, end_dp=end, axis=None, pushoff_df=True, timestamps=None, xz_data=None):
        """
        Originally def step_detect(self)
        Detects the steps within the accelerometer data. Based on this paper:
        https://ris.utwente.nl/ws/portalfiles/portal/6643607/00064463.pdf
        """
        # state space controller function definitions
        def get_pushoff_sigs(step_obj, peaks=None, quiet=False):
            """TODO: This needs the inputs from the step_obj used here pushoff time, freq, puhoff len, etc.
            Creates average pushoff dataframe that is used to find pushoff data
            ---
            Parameters
            ---
            step_obj -> this is the output from StepDetection - trying to figure out where it comes form

            """
            pushoff_sig_list = []
            pushoff_len = step_obj.pushoff_time * step_obj.freq

            if not peaks:
                peaks = np.array(step_obj.step_indices) + pushoff_len
            for i, peak_ind in tqdm(enumerate(peaks), desc="Generating pushoff average", total=len(peaks),
                                    disable=quiet,
                                    leave=False):
                pushoff_sig = step_obj.data[int(peak_ind - pushoff_len):int(peak_ind)]
                pushoff_sig_list.append(pushoff_sig)

            return np.array(pushoff_sig_list)

        def export_steps(detect_arr=None, state_arr=None, timestamps=None, step_indices=None, start_dp=None,
                         pushoff_time=None, foot_down_time=None, success=True):
            """
            Export steps into a dataframe - this includes all potential push-offs and the state that they fail on
            ---
            Parameter
            ---

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
            avg_speed = [np.mean(xz_data[i:i + int(lengths * freq)]) * 9.81 * lengths for i, lengths in
                         zip(step_indices, step_durations)]

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
                'avg_speed': avg_speed
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

        def get_pushoff_stats(accel_path, start_end_times=[(-1, -1)], axis=None, quiet=True):
            """
            Calculate push-off  for step detection - uses default if many steps are not found. (20 minimum)

            This function creates a pushoff_df when there is no pushoff_df defined. Essentially it will use
            the current pushoff_df to find steps, then create a new pushoff_df from those steps that were found.

            We will define the pushoff_df in the acc_find_steps function and therefore
            TODO: Create a new function that can find pushoffs from raw data without calling acc_step_detect
            ---
            Parameters
            ---
            accel_path -> accelerometer path
            start_end_times -> start and end times of known walking
            axis -> axis detection
            ---
            Return
            ---
            pushoff_df -> dataframe with push off slopes
            """

            pushoff_sig_list = []
            swingdown_times = []
            swingup_times = []
            heelstrike_values = []
            dir_path = os.path.dirname(os.path.realpath(__file__))
            pushoff_df = pd.read_csv(os.path.join(dir_path, 'data', 'pushoff_OND07_left.csv'))

            for start, end in start_end_times:
                # TODO: Where does StepDetection come from...is it step_detect? This is just a step object - so it has all the parameters within it
                # step = StepDetection(accel_path_or_obj=accel_path, label='get_pushoff_stats',
                #                      axis=axis, start=start, end=end, quiet=True, pushoff_df=pushoff_df)
                pushoff_sig = get_pushoff_sigs(step, quiet=quiet)
                step_summary = export_steps(step)
                toe_offs = step_summary.loc[step_summary['step_state'] == 'success', 'swing_start_time']
                mid_swings = step_summary.loc[step_summary['step_state'] == 'success', 'mid_swing_time']
                heel_strikes = step_summary.loc[step_summary['step_state'] == 'success', 'heel_strike_time']
                step_indices = step_summary.loc[
                                   step_summary['step_state'] == 'success', 'step_index'] - step.start_dp

                mid_swing_indices = step_indices + (
                        step.pushoff_time + (mid_swings - toe_offs).dt.total_seconds()) * step.freq

                if len(pushoff_sig) == 0:
                    print('WARNING: No steps found (start=%s, end=%s)' % (str(start), str(end)))
                    continue
                pushoff_sig_list.append(pushoff_sig)
                swingdown_times.append((mid_swings - toe_offs).dt.total_seconds())
                swingup_times.append((heel_strikes - mid_swings).dt.total_seconds())
                heelstrike_values.append(
                    [np.min(step.heel_strike_detect(int(ms_ind))) for ms_ind in mid_swing_indices])

            if len(pushoff_sig_list) == 0:
                return None

            pushoff_sig_list = np.concatenate(pushoff_sig_list)
            swingdown_times = np.concatenate(swingdown_times)
            swingup_times = np.concatenate(swingup_times)
            heelstrike_values = np.concatenate(heelstrike_values)
            po_avg_sig = np.mean(pushoff_sig_list, axis=0)
            po_std_sig = np.std(pushoff_sig_list, axis=0)
            po_max_sig = np.max(pushoff_sig_list, axis=0)
            po_min_sig = np.min(pushoff_sig_list, axis=0)

            sdown_mean = np.nanmean(swingdown_times)
            sdown_std = np.nanstd(swingdown_times)
            sup_mean = np.nanmean(swingup_times)
            sup_std = np.nanstd(swingup_times)
            hs_mean = np.nanmean(sorted(heelstrike_values, reverse=True)[:len(heelstrike_values) // 4])
            hs_std = np.nanstd(sorted(heelstrike_values, reverse=True)[:len(heelstrike_values) // 4])

            if len(pushoff_sig_list) < 20:
                print('WARNING: less than 20 steps used for pushoff DF, using default pushoff_df')
                return pushoff_df

            pushoff_df = pd.DataFrame(
                {'avg': po_avg_sig, 'std': po_std_sig, 'max': po_max_sig, 'min': po_min_sig,
                 'swing_down_mean': sdown_mean, 'swing_down_std': sdown_std,
                 'swing_up_mean': sup_mean, 'swing_up_std': sup_std,
                 'heel_strike_mean': hs_mean, 'heel_strike_std': hs_std})

            return pushoff_df

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

        def push_off_detection(data=None, pushoff_df=None, push_off_threshold=None, freq=None):
            """
            Detects the steps based on the pushoff_df
            """
            pushoff_avg = pushoff_df['avg']

            cc_list = window_correlate(data, pushoff_avg)

            # TODO: DISTANCE CAN BE ADJUSTED FOR THE LENGTH OF ONE STEP RIGHT NOW ASSUMPTION IS THAT A PERSON CANT TAKE 2 STEPS WITHIN 0.5s
            pushoff_ind, _ = find_peaks(cc_list, height=push_off_threshold, distance=max(0.2 * freq, 1))

            return pushoff_ind

        def mid_swing_peak_detect(data=None, pushoff_ind=None, swing_phase_time=None):
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

        def swing_detect(self, pushoff_ind, mid_swing_ind):
            """
            Detects swings (either up or down) given a starting index (window_ind).
            Swing duration is preset.
            """
            # swinging down
            detect_window = self.data[pushoff_ind:mid_swing_ind]
            swing_len = mid_swing_ind - pushoff_ind
            swing_down_sig = -np.arange(swing_len) + swing_len / 2 + np.mean(detect_window)

            # swinging up
            swing_up_detect = int(self.freq * self.swing_up_detect_time)  # length to check for swing
            swing_up_detect_window = self.data[mid_swing_ind:mid_swing_ind + swing_up_detect]
            swing_up_sig = -(-np.arange(swing_up_detect) + swing_up_detect / 2 + np.mean(detect_window))

            swing_down_cc = [np.corrcoef(detect_window, swing_down_sig)[0, 1]] if detect_window.shape[0] > 1 else [
                0]
            swing_up_cc = [np.corrcoef(swing_up_detect_window, swing_up_sig)[0, 1]] if swing_up_detect_window.shape[
                                                                                           0] > 1 else [0]

            return (swing_down_cc, swing_up_cc)

        def heel_strike_detect(data=None, heel_strike_detect_time=None, window_ind=None, freq=None):
            """
            Detects a heel strike based on the change in acceleration over time.
            """
            type(freq)
            type(heel_strike_detect_time)
            heel_detect = int(freq * heel_strike_detect_time)
            detect_window = data[window_ind:window_ind + heel_detect]
            accel_t_plus1 = np.append(
                detect_window[1:detect_window.size], detect_window[-1])
            accel_t_minus1 = np.insert(detect_window[:-1], 0, detect_window[0])
            accel_derivative = (accel_t_plus1 - accel_t_minus1) / (2 / freq)

            return accel_derivative

        def plot(self, return_plt=False):
            """
            Plots the accelerometer data, the states detected, and the detected pushoffs that were eliminated
            """

            dp_range = np.arange(self.start_dp, self.end_dp)

            ax1 = plt.subplot(311)
            ax1.set_title('Accelerometer Data')
            plt.plot(dp_range, self.data, 'r-')
            # plt.plot(dp_range[self.swing_peaks], self.data[self.swing_peaks], 'bo')
            plt.grid(True)

            ax2 = plt.subplot(312, sharex=ax1)
            states_legend = ['stance', 'pushoff',
                             'swing down', 'swing up', 'heel strike']
            ax2.set_title('States of Steps in Accelerometer Data')
            ax2.set_yticks(np.arange(len(states_legend)))
            ax2.set_yticklabels(states_legend)
            # ax2.legend([0,1,2,3,4], ['stance', 'pushoff', 'swing down', 'swing up', 'heel strike'])
            plt.plot(dp_range, self.state_arr, "b-")
            plt.grid(True)

            ax3 = plt.subplot(313, sharex=ax1)
            ax3.set_title('Push off signals filtered out by SSC')
            filtered_legend = ['', 'swing down', 'swing up', 'heel strike',
                               'next pushoff too close', 'pushoff mean too far', 'mid_swing_peak']
            ax3.set_yticks(np.arange(len(filtered_legend)))
            ax3.set_yticklabels(filtered_legend)
            plt.plot(dp_range, self.detect_arr, "go")
            plt.grid(True)

            plt.tight_layout()
            if not return_plt:
                plt.show()
            else:
                return plt

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
        freq = device.signal_headers[axis]['sample_rate']
        pushoff_len = int(pushoff_time * freq)
        states = {1: 'stance', 2: 'push-off',
                  3: 'swing-up', 4: 'swing-down', 5: 'footdown'}
        state = states[1]
        #defining step pushoff thresholds
        #this says if pushoff_df is None then run "get_push_off_stats" if pushoff_df is defined then pushoff_df=pushoff_df
        if pushoff_df == True:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            pushoff_df = pd.read_csv(os.path.join(dir_path, 'data', 'pushoff_OND07_left.csv'))
        else:
            pushoff_df = get_pushoff_stats(data, start_end_times=[(start_dp, end_dp)], axis=axis)
            # pushoff_df = get_pushoff_stats(start_end_times=[(-1, -1)])
        swing_peaks = []

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
        #initialize
        end_i = None
        step_indices = []
        step_lengths = []
        #run
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

            # midswing peak detection -> mid_swing_peak_detect(data=None, pushoff_ind=None, swing_phase_time=None):
            mid_swing_i = mid_swing_peak_detect(data, i, swing_phase_time)
            if mid_swing_i is None:
                detects['mid_swing_peak'].append(i - 1)
                continue

            # # swing down, state = 2
            # sdown_cc, sup_cc = self.swing_detect(i, mid_swing_i)
            # if not max(sdown_cc) > self.swing_threshold:
            #     detects['swing_down'].append(i - 1)
            #     continue
            #
            # # swing up, state = 3
            # if not max(sup_cc) > self.swing_threshold:
            #     detects['swing_up'].append(i - 1)
            #     continue

            # heel-strike, state = 4 -> def heel_strike_detect(data=None, heel_strike_detect_time=None, window_ind=None, freq=None):
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

        steps_df = export_steps(detect_arr, state_arr, timestamps, step_indices, start_dp, pushoff_time, foot_down_time)

        return steps_df

    def detect_steps_gyro(device=None, data=None):

        #define functions for gyroscope step detection
        def create_timestamps(data_start_time, data_len, fs, start_time=None, end_time=None):

            start_time = data_start_time if start_time is None else start_time
            end_time = data_start_time + timedelta(seconds=(data_len / fs)) if end_time is None else end_time

            indexes = [int((start_time - data_start_time).total_seconds() * fs),
                       int((end_time - data_start_time).total_seconds() * fs)]
            length = indexes[1]-indexes[0]
            timestamps = pd.date_range(start=start_time, periods=length, freq=f'{1/fs}S')

            return timestamps, indexes

        def bw_filter(data, fs, fc,  order):
            # #dev
            # data = gyro_data[indexes[0]:indexes[1]]
            # plt.plot(data)
            #
            # order = 5
            # fs = sample_rate
            # fc = 5

            # scipy.signal.filtfilt(b, a, x, axis=- 1, padtype='odd', padlen=None, method='pad', irlen=None)
            b, a = butter(N=order, Wn=fc, btype='low', output='ba', fs=fs)
            filtered_data = filtfilt(b, a, data)
            # plt.plot(filtered_data)

            return filtered_data


        def find_adaptive_thresh(data):
            # this function find the adaptive threshold (on pre-processed data) according to steps found in
            # Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014). Real-world gyroscope-based gait event detection
            # and gait feature extraction.

            # 2a: Calculate derivative of signal
            data_2d = np.diff(data)/0.02
            # another differential method
            # diff = np.zeros(len(data_f) - 1)
            # for idx, value in enumerate(diff):
            #         diff[idx] = (data_f[(idx + 1)] - data_f[(idx)]) /(1/sampl_freqs[axis])

            # 2b: Calculate adaptive threshold from 10 max peaks in signal derivative
            # note this threshold is applied to pre-procesed data for peak detection

            thresh = np.mean(data[np.argpartition(data_2d, 10)[:10]])*0.2
            if thresh > 40:
                pass
            else:
                thresh = 40

            return thresh


        def remove_single_step_bouts(df, steps_length=2):
            sum_df = df.groupby(['Bout_number']).count()
            sum_df.columns = ['Step_number', 'Step_index', 'Peak_times']

            sum_df.drop(sum_df[sum_df.Step_number < steps_length].index, inplace=True)
            bout_index = sum_df.index

            df= df[df.Bout_number.isin(bout_index)]
            df.reset_index(inplace=True, drop=True)
            df = renumber_bouts(df)
            #df.iloc[:, 0] = np.arange(1, len(df) + 1)
            return df


        def renumber_bouts(df):
            orig_bouts = df.Bout_number
            num = 1
            for i in range(len(orig_bouts)):
                if i == 0:
                    df.loc[i,2] = num
                else:
                    if orig_bouts[i] > orig_bouts[i - 1]:
                        num = num + 1
                        df.loc[i,2] = num
                    else:
                        df.loc[i,2] = num
            df.drop('Bout_number', inplace=True, axis=1)
            df.columns= ['Step','Step_index','Peak_times','Bout_number']

            return df


        def get_bouts_data(df):
            # dev
            # df = bout_events_df

            bout_list = df['Bout_number'].unique()
            bout_df = pd.DataFrame(columns=['Bout_number', 'Step_count', 'Start_time', 'End_time', 'Start_idx', 'End_idx'])
            for count, val in enumerate(bout_list):
                # print(f'count:{count}, value:{val}')
                temp = df[df['Bout_number'] == bout_list[count]]
                # print(temp)
                step_count = len(temp)
                start_time = np.min(temp['Peak_times'])
                end_time = np.max(temp['Peak_times'])
                start_ind = np.min(temp['Step_index'])
                end_ind = np.max(temp['Step_index'])
                # cadence = step_count/((end_ind-start_ind)*(1/fs)/60)
                data = pd.DataFrame([[count + 1,  step_count, start_time, end_time, start_ind, end_ind]], columns=bout_df.columns)    # , "Cadence":cadence}
                bout_df = pd.concat([bout_df, data], ignore_index=True)

            return bout_df


        def gyro_step_indexes(data, gait_bouts_df=None, sample_freq=None, min_swing_t=0.250, max_swing_t=0.800):
            # # dev
            # data = gyro_data[indexes[0]:indexes[1]]
            # gait_bouts_df = gait_bouts_df
            # sample_freq = sample_rate
            # min_swing_ms = 250
            # max_swing_ms = 800

            # create index values for min and maximum swing time
            min_swing_idx = sample_freq * min_swing_t
            max_swing_idx = sample_freq * max_swing_t

            # 1: low-pass at 5hz
            # data=bw_filter(data=data, fs=sample_freq, fc=5, order=5)
            # plt.plot(data)
            # 2: Adaptive threshold as per:
            # B.R. Greene, et al., ”Adaptive estimation of temporal gait parameters using body-worn gyroscopes,”
            # Proc. IEEE Eng. Med. Bio. Soc. (EMBC 2011), pp. 1296-1299, 2010
            # and outlined in Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014)
            # ic = []
            # tc = []
            # frequency = []
            peaks = []
            gyro_mean = []
            thresholds = []
            mnf = []
            mdf = []

            initial_contacts = []
            terminal_contacts = []
            # for i in range(len(gait_bouts_df)):
            #         # In each iteration, add an empty list to the main list
            #         initial_contacts.append([])
            #         terminal_contacts.append([])

            for row in gait_bouts_df.itertuples():

                # print(row)
                gyro_z_mean = np.mean(data[int(row[5]):int(row[6])])
                gyro_mean.append(gyro_z_mean)
                th2 = 0.8 * (1 / (sum(data[int(row[5]):int(row[6])] > gyro_z_mean)))
                if th2 > 40:
                    pass
                else:
                    th2 = 40
                thresholds.append(th2)

                peak_idx, _ = find_peaks(x=np.concatenate((np.zeros(1), data[int(row[5]):int(row[6])], np.zeros(1))),
                                            height=th2, distance=sample_freq * 0.5)
                # correct peak_idx
                peak_idx = peak_idx - 1 + int(row[5])
                # print(peak_idx)
                peaks.append(peak_idx)

                # plt.plot((data[int(row[5]):int(row[6])]))
                # plt.axhline(th2, ls='--', c='red')
                # plt.scatter(x=peak_idx, y=data[int(row[5]):int(row[6])][peak_idx], marker='x', color='orange')

                ics = []
                tcs = []

                for i in range(len(peak_idx)):
                    # print(i)
                    window_len = math.floor(0.4 * sample_freq)
                    # plt.plot(data[int(peak_idx[i]-window_len):peak_idx[i+1]])

                    # adaptation to the Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014) description of this algorithm
                    # final contacts (toe-down or flat foot) doesn't always pick off the middle peak, specifically with spikes at initail contact
                    # fc, _ = sp.find_peaks(data[int(row[5]):int(row[6])][peak_idx[i]:peak_idx[i+1]], distance=len(data[int(row[5]):int(row[6])])*0.5)
                    # print(fc)
                    # as per modification above - window_len is described here as half the distance between the two peaks

                    tc = np.argmin(data[int(peak_idx[i] - window_len):peak_idx[i]]) + peak_idx[i] - window_len
                    ic = np.argmin(data[peak_idx[i]:int(peak_idx[i] + window_len)]) + peak_idx[i]
                    # tc = np.argmin(data[int(peak_idx[i] - window_len):peak_idx[i]]) + int(row[5]) + (
                    #                 (peak_idx[i] - window_len) - int(row[5]))
                    # ic = np.argmin(data[peak_idx[i]:int(peak_idx[i] + window_len)]) + int(row[5]) + (
                    #                 peak_idx[i] - int(row[5]))
                    # print(tc, peak_idx[i], ic)

                    if max_swing_idx < ic - tc < min_swing_idx:
                        continue
                    else:
                        ics.insert(i, ic)
                        tcs.insert(i, tc)

                    # plt.plot(data[peak_idx[0]-100:peak_idx[-1]+100])
                    # plt.scatter(x=np.array(tcs)-row[5]+100, y=data[tcs], marker='o', color='green')
                    # plt.scatter(x=peak_idx-row[5]+100, y=data[peak_idx], marker='x', color='orange')
                    # plt.scatter(x=np.array(ics)-row[5]+100, y=data[ics], marker='o', color='red')

                # initial_contacts[int(row[0])].insert(0,ics)
                # terminal_contacts[int(row[0])].insert(0,tcs)
                initial_contacts.insert(int(row[0]), ics)
                terminal_contacts.insert(int(row[0]), tcs)

                # frequency of bouts
                freqs, psd = welch(data[int(row[5]):int(row[6])], fs=sample_freq)
                # plt.figure(figsize=(5, 4))
                # plt.plot(freqs, psd)
                # plt.title('PSD: power spectral density')
                # plt.xlabel('Frequency')
                # plt.ylabel('Power')
                # plt.tight_layout()

                temp_mnf = sum(freqs * psd) / sum(psd)
                # print(MNF)
                temp_mdf = freqs[np.argmin(np.abs(np.cumsum(psd) - (0.5 * sum(psd))))]
                # print(MDF)
                mnf.append(temp_mnf)
                mdf.append(temp_mdf)

            gait_bouts_df['Gryo_z_mean'] = gyro_mean
            gait_bouts_df['Threshold'] = thresholds
            gait_bouts_df['MS_peaks'] = peaks
            gait_bouts_df['Initial_contacts'] = initial_contacts
            gait_bouts_df['Terminal_contacts'] = terminal_contacts
            gait_bouts_df['Mean power freq'] = mnf
            gait_bouts_df['Median power freq'] = mdf

            # need to correct start idx and end idx (plus timestamps

            return gait_bouts_df


        def get_gait_bouts(data, sample_freq,  timestamps, break_sec=2, bout_steps=3, start_ind=0, end_ind=None):

            # crop data
            data = data[start_ind:end_ind]

            # low pass filter at 3 hz
            # 1: LP filter data at 3 Hz
            lf_data = bw_filter(data=data, fs=sample_freq, fc=3, order=5)

            # 2: Calculate adaptive threshold
            th1 = find_adaptive_thresh(data=lf_data)

            # 3: Group MS peaks to identify gait events
            # how far should peaks be apart?

            # identify peaks above calculated threshold
            idx_peaks, peak_hghts = find_peaks(x=data, height=th1, distance=40)  # at 50 samples/sec; 5 samples = 100 ms/0.1s; 10 samples = 200 ms/0.2s
            peak_heights = peak_hghts.get('peak_heights')

            # create start of idx_peaks
            # determine indices between peaks
            peaks_diff = np.diff(idx_peaks)  # difference between index of peaks - so number datapoints between peaks

            # setting up bout counting
            ge_break_ind = break_sec * sample_freq  # threshold for indices between peaks that causes a new bout
            # count_ge_diff = (peaks_diff > ge_break_ind).sum() #this gives be the number of gait bouts that occur based on break_time difference between peaks also the len of ind_ge_diff below
            bool_diff = peaks_diff > ge_break_ind  # creates boolean array of where differences are greater than break and indicated which gaps will be different bouts
            # but doesn't point to peak_index - but to the index of the peaks_diff array represented as the ind_gait event_diff
            ind_ge_diff = [i for i, x in enumerate(bool_diff) if x]  # return indices array True in boolean
            #
            # for count, val in enumerate(idx_peaks):
            #         if idx_peaks[count+1]-idx_peaks[count] <ge_break_ind:

            # assign gait bout event label
            bouts = np.zeros(len(idx_peaks))

            # for loop that counts the iterations of ind_ge_diff and counts
            for count, x in enumerate(ind_ge_diff):
                # print(f'Count: {count}, Index_ge_diff:{idx_peaks[count]}, Value:{x}')
                if count < len(ind_ge_diff) - 1:
                    if x == 0:
                        bouts[count] = 1
                        continue
                    elif ind_ge_diff[count] - ind_ge_diff[count + 1] == 1:
                        bouts[count] = count + 1
                        continue
                    else:
                        if count == 0:
                            bouts[:(ind_ge_diff[count+1])] = count + 1
                        else:
                            bouts[(ind_ge_diff[count-1]+1):(ind_ge_diff[count+1]+1)] = count + 1
                elif count == len(ind_ge_diff)-1:
                    bouts[(ind_ge_diff[count - 1] + 1):ind_ge_diff[count]+1] = count + 1
                    bouts[ind_ge_diff[count] + 1:] = count + 2

            # print(bouts)

            step_count = range(1, len(idx_peaks) + 1)
            step_events_df = pd.DataFrame({'Step': step_count, 'Step_index': idx_peaks, 'Bout_number': bouts})

            # 4: Ensure left and right occurrence of gait events
            # single IMU only - no need for this step atm

            # 5a: Get timestamps for all steps for output with dataframe
            step_events_df['Step_timestamp'] = timestamps[step_events_df['Step_index']]

            # 5b: Remove gait events that were one or less steps
            gait_bouts_df = remove_single_step_bouts(df=step_events_df, steps_length=bout_steps)

            # 5c: output bouts_df of bout data (step counts, start/end times, start/end ind)"
            gait_bouts_df = get_bouts_data(df=gait_bouts_df)

            # renumber the bouts in steps_df
            step_events_df['Bout_number'] = 0
            for i in range(len(gait_bouts_df)):
                bool = (step_events_df.Step_index >= gait_bouts_df.Start_idx[i]) & (
                            step_events_df.Step_index <= gait_bouts_df.End_idx[i])
                idx = step_events_df.index[bool]
                step_events_df.Bout_number.iloc[idx] = gait_bouts_df.Bout_number[i]
            # step_events_df.Bout_number.replace(0, np.nan, inplace=True) #if you want non-bouts to show up at NaN

            # 6a: iterate through gait bouts to find gait bout tabular data: mean cycle time, stance and swing times,
            # frequency analysis
            # gait_bouts_df = gyro_step_indexes(data=data, gait_bouts_df=gait_bouts_df, sample_freq=sample_freq)

            # 6b: adjust start_idx to equal first ic and vv for terminal

            # 6c: need to look at th3 th4 to determine whether we should keep or reject step

            # 7: add a plotting function

            step_events_df.columns = ['step_number', 'step_index','bout_number', 'step_timestamp']

            return step_events_df, peak_heights #gait_bouts_df,

        def gait_stats(bouts, type='daily', single_leg=False):

            bouts['date'] = pd.to_datetime(bouts['start_time']).dt.date
            bouts['duration'] = [round((x['end_time'] - x['start_time']).total_seconds()) for i, x in bouts.iterrows()]

            # if only steps from one leg then double step counts
            bouts['step_count'] = bouts['step_count'] * 2 if single_leg else bouts['step_count']

            if type == 'daily':

                gait_stats = pd.DataFrame(columns=['day_num', 'date', 'type', 'longest_bout_time', 'longest_bout_steps',
                                                   'bouts_over_3min', 'total_steps'])

                day_num = 1

                for date, group_df in bouts.groupby('date'):

                    day_gait_stats = pd.DataFrame([[day_num, date, type, group_df['duration'].max(),
                                                round(group_df['step_count'].max()),
                                                group_df.loc[group_df['duration'] > 180].shape[0],
                                                round(group_df['step_count'].sum())]], columns=gait_stats.columns)

                    gait_stats = pd.concat([gait_stats, day_gait_stats], ignore_index=True)

                    day_num += 1

            else:
                print('Invalid type selected.')

            return gait_stats

        #define parameters
        freq = int(device.signal_headers[0]['sample_rate'])
        ## get signal labels
        index_dict = {"gyro_x": device.get_signal_index('Gyroscope x'),
                      "gyro_y": device.get_signal_index('Gyroscope y'),
                      "gyro_z": device.get_signal_index('Gyroscope z'),
                      "accel_x": device.get_signal_index('Accelerometer x'),
                      "accel_y": device.get_signal_index('Accelerometer y'),
                      "accel_z": device.get_signal_index('Accelerometer z')}
        ##get signal frequnecies needed for step detection
        gyro_freq = int(device.signal_headers[index_dict['gyro_x']]['sample_rate'])
        accel_freq = int(device.signal_headers[index_dict['accel_x']]['sample_rate'])
        ##get start stamp
        data_start_time = device.header["start_datetime"] if start is None else start
        ## get data for gyro step detection
        gyro_data = device.signals[index_dict['gyro_z']]
        data_len = len(gyro_data)

        #run
        timestamps, indexes = create_timestamps(data_start_time, data_len, fs=gyro_freq, start_time=None, end_time=None)

        steps_df, peak_heights = get_gait_bouts(data=gyro_data, sample_freq=gyro_freq, timestamps=timestamps, break_sec=2, bout_steps=3,
                                                      start_ind=0, end_ind=None)

        return steps_df #TODO - remove the bouting and only output step_num, step_index, _step_timestamp to feed into steps_df

    if device.header['device_type'] == 'GNAC':
        print(f"Device set: {device.header['device_type']} detecting steps using accelerometer.")
        """
        """
        #acc_step_detect_ssc(data=None, start_dp=1, end_dp=-1, pushoff_df=None)
        steps_df = detect_steps_ssc(device = ankle_acc, data= data,  start_dp=100000, end_dp=100000, axis=axis, pushoff_df=True, timestamps=timestamps, xz_data=xz_data)

    elif device.header['device_type'] == 'AXV6':
        print(f"Device set: {device.header['device_type']} detecting steps using gryoscope.")

        steps_df = detect_steps_gyro(device=None)


    else:
        print("Device not defined. State space step detector not run.")

    return steps_df


# Walkingbouts declassed
def get_walking_bouts(left_steps_df=None, right_steps_df=None, right_device=None, left_device=None, duration_sec=15, bout_num_df=None,
                      legacy_alg=False, left_kwargs={}, right_kwargs={}):
    """

    """

    def find_dp(path, duration_sec, timestamp_str=None, axis=1):
        """
        Gets start and end time based on a timestamp and duration_sec (# data points)
        """
        accel_file = pyedflib.EdfReader(path)
        time_delta = pd.to_timedelta(
            1 / accel_file.getSampleFrequency(axis), unit='s')
        start = 0
        if timestamp_str:
            start = int((pd.to_datetime(timestamp_str) -
                         accel_file.getStartdatetime()) / time_delta)
        end = int(start + pd.to_timedelta(duration_sec, unit='s') / time_delta)
        accel_file.close()

        return start, end

    def identify_bouts_one(steps_df, freq):

        steps = steps_df['step_index']  # step_detector.step_indices
        timestamps = steps_df['step_timestamp']  # step_detector.timestamps[steps]
        step_durations = steps_df['step_duration']  # step_detector.step_lengths

        freq=int(freq)

        steps_df = pd.DataFrame({'step_index': steps, 'timestamp': timestamps, 'step_duration': step_durations})
        steps_df = steps_df.sort_values(by=['step_index'], ignore_index=True)

        # assumes Hz are the same
        bout_dict = {'start': [], 'end': [], 'number_steps': [], 'start_time': [], 'end_time': []}
        if steps_df.empty:
            return pd.DataFrame(bout_dict)
        start_step = steps_df.iloc[0]  # start of bout step
        curr_step = steps_df.iloc[0]
        step_count = 1
        next_steps = None

        while curr_step is not None:
            # Assumes steps are not empty and finds the next step after the current step
            termination_bout_window = pd.Timedelta(15, unit='sec') if next_steps is None else pd.Timedelta(10,
                                                                                                           unit='sec')
            next_steps = steps_df.loc[(steps_df['timestamp'] <= termination_bout_window + curr_step['timestamp'])
                                      & (steps_df['timestamp'] > curr_step['timestamp'])]

            if not next_steps.empty:
                curr_step = next_steps.iloc[0]
                step_count += 1
            else:
                # stores bout
                if step_count >= 3:
                    print(curr_step)
                    start_ind = start_step['step_index']
                    end_ind = curr_step['step_index'] + curr_step['step_duration']
                    bout_dict['start'].append(start_ind)
                    bout_dict['end'].append(end_ind)
                    bout_dict['number_steps'].append(step_count)
                    bout_dict['start_time'].append(start_step['timestamp'])
                    bout_dict['end_time'].append(
                        curr_step['timestamp'] + pd.Timedelta(curr_step['step_duration'] / freq, unit='sec'))

                # resets state and creates new bout
                step_count = 1
                next_curr_steps = steps_df.loc[steps_df['timestamp'] > curr_step['timestamp']]
                curr_step = next_curr_steps.iloc[0] if not next_curr_steps.empty else None
                start_step = curr_step
                next_steps = None

        bout_num_df = pd.DataFrame(bout_dict)
        return bout_num_df

    def find_overlapping_times(left_bouts, right_bouts):
        # merge based on step index
        export_dict = {'start': [], 'end': [], 'number_steps': [], 'start_time': [], 'end_time': []}
        all_bouts = pd.concat([left_bouts, right_bouts])
        all_bouts = all_bouts.sort_values(by=['start_time'], ignore_index=True)
        all_bouts['overlaps'] = (all_bouts['start_time'] < all_bouts['end_time'].shift()) & (
                all_bouts['start_time'].shift() < all_bouts['end_time'])
        all_bouts['intersect_id'] = (((all_bouts['overlaps'].shift(-1) == True) & (all_bouts['overlaps'] == False)) |
                                     ((all_bouts['overlaps'].shift() == True) & (
                                             all_bouts['overlaps'] == False))).cumsum()

        for intersect_id, intersect in all_bouts.groupby('intersect_id'):
            # if there are no overlaps i want to iterate each individual bout
            if not intersect['overlaps'].any():
                for i, row in intersect.iterrows():
                    export_dict['start'].append(row['start'])
                    export_dict['end'].append(row['end'])
                    export_dict['number_steps'].append(row['number_steps'])
                    export_dict['start_time'].append(row['start_time'])
                    export_dict['end_time'].append(row['end_time'])
            else:
                export_dict['start'].append(intersect['start'].min())
                export_dict['end'].append(intersect['end'].max())
                export_dict['number_steps'].append(intersect['number_steps'].sum())
                export_dict['start_time'].append(intersect['start_time'].min())
                export_dict['end_time'].append(intersect['end_time'].max())

        df = pd.DataFrame(export_dict)
        df = df.sort_values(by=['start_time'], ignore_index=True)
        df['overlaps'] = (df['start_time'] < df['end_time'].shift()) & (
                df['start_time'].shift() < df['end_time'])

        # if there are no overlaps
        if not df['overlaps'].any():
            df = df.drop(['overlaps'], axis=1)
            return df
        else:
            return find_overlapping_times(df, pd.DataFrame())  # makes an empty dataframe to compare

    def identify_bouts(left_stepdetector, right_stepdetector, freq):
        """
        Identifies the bouts within the left and right acceleromter datas.
        The algorithm finds bouts that have 3 bilateral steps within a 15 second window
        """
        left_step_i = left_stepdetector.step_indices
        right_step_i = right_stepdetector.step_indices
        assert left_stepdetector.freq == right_stepdetector.freq
        freq = left_stepdetector.freq

        # merge into one list
        steps = np.concatenate([left_step_i, right_step_i])
        step_lengths = np.concatenate(left_stepdetector.step_lengths[left_step_i],
                                      right_stepdetector.step_lengths[right_step_i])
        foot = np.concatenate([['L'] * len(left_step_i), ['R'] * len(right_step_i)])
        timestamps = np.concatenate(
            [left_stepdetector.timestamps[left_step_i], right_stepdetector.timestamps[right_step_i]])
        steps_df = pd.DataFrame(
            {'step_index': steps, 'foot': foot, 'timestamp': timestamps, 'step_length': step_lengths})
        steps_df = steps_df.sort_values(by=['step_index'], ignore_index=True)

        # assumes Hz are the same
        bout_dict = {'start': [], 'end': [], 'bilateral_steps': [], 'start_time': [], 'end_time': []}
        if steps_df.empty:
            return pd.DataFrame(bout_dict)
        start_step = steps_df.iloc[0]  # start of bout step
        curr_step = steps_df.iloc[0]
        bilateral_count = 0
        step_count = 1
        next_steps = None

        if steps_df.empty:
            return pd.DataFrame(bout_dict)

        while curr_step is not None:
            # Assumes steps are not empty
            termination_bout_window = pd.Timedelta(15, unit='sec') if next_steps is None else pd.Timedelta(10,
                                                                                                           unit='sec')
            next_steps = steps_df.loc[(steps_df['foot'] != curr_step['foot'])
                                      & (steps_df['timestamp'] <= termination_bout_window + curr_step['timestamp'])
                                      & (steps_df['timestamp'] > curr_step['timestamp'])]

            if not next_steps.empty:
                # iterate to next step
                curr_step = next_steps.iloc[0]
                bilateral_count += 1 if curr_step['foot'] != start_step['foot'] else 0
                step_count += 1
            else:
                # store/reset variables. begin new bout
                if bilateral_count >= 3:
                    start_ind = start_step['step_index']
                    end_ind = curr_step['step_index'] + curr_step['step_length']
                    bout_dict['start'].append(start_ind)
                    bout_dict['end'].append(end_ind)
                    bout_dict['number_steps'].append(step_count)
                    bout_dict['bilateral_steps'].append(bilateral_count)
                    bout_dict['start_time'].append(start_step['timestamp'])
                    bout_dict['end_time'].append(
                        curr_step['timestamp'] + pd.Timedelta(curr_step['step_length'] / freq, unit='sec'))

                bilateral_count = 0
                step_count = 1
                next_curr_steps = steps_df.loc[steps_df['timestamp'] > curr_step['timestamp']]
                curr_step = next_curr_steps.iloc[0] if not next_curr_steps.empty else None
                start_step = curr_step
                next_steps = None

        bout_num_df = pd.DataFrame(bout_dict)
        bout_num_df['left_cycle_count'] = [len(steps_df.loc[(steps_df['foot'] == 'L')
                                                            & (steps_df['step_index'] >= bout_num_df.iloc[i]['start'])
                                                            & (steps_df['step_index'] <= bout_num_df.iloc[i]['end'])])
                                           for i in bout_num_df.index]
        bout_num_df['right_cycle_count'] = [len(steps_df.loc[(steps_df['foot'] == 'R')
                                                             & (steps_df['step_index'] >= bout_num_df.iloc[i]['start'])
                                                             & (steps_df['step_index'] <= bout_num_df.iloc[i]['end'])])
                                            for i in bout_num_df.index]

        return bout_num_df

    def export_bout_steps(bout_num_df, left_steps_df, right_steps_df):
        bout_steps = []
        for i, row in bout_num_df.iterrows():
            start = row['start_time'] - pd.Timedelta(1, unit='sec')
            end = row['end_time'] + pd.Timedelta(1, unit='sec')

            left_bout_step_df = left_steps_df.loc[
                (left_steps_df['step_timestamp'] > start) & (left_steps_df['step_timestamp'] < end)]
            right_bout_step_df = right_steps_df.loc[
                (right_steps_df['step_timestamp'] > start) & (right_steps_df['step_timestamp'] < end)]

            bout_step_df = pd.concat([left_bout_step_df, right_bout_step_df])
            bout_step_df['gait_bout_num'] = i + 1
            bout_steps.append(bout_step_df)

        if len(bout_steps) == 0:
            return pd.DataFrame()

        bout_step_summary = pd.concat(bout_steps)
        bout_step_summary.sort_values(by=['gait_bout_num', 'step_timestamp'])
        bout_step_summary = bout_step_summary.reset_index()
        bout_step_summary['step_num'] = bout_step_summary.index + 1

        return bout_step_summary

    def verbose_bout_output(bout_output, bout_step_summary):

        for i, bout in bout_step_summary.groupby('gait_bout_num'):
            bout_output.loc[bout_output['gait_bout_num'] == i, 'success_left_step_count'] = \
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'left')].shape[0]
            bout_output.loc[bout_output['gait_bout_num'] == i, 'success_right_step_count'] = \
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'right')].shape[0]

            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_left_heel_strike_accel'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'left'), 'heel_strike_accel'])
            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_right_heel_strike_accel'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'right'), 'heel_strike_accel'])

            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_left_mid_swing_accel'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'left'), 'mid_swing_accel'])
            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_right_mid_swing_accel'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'right'), 'mid_swing_accel'])

            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_left_swing_start_accel'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'left'), 'swing_start_accel'])
            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_right_swing_start_accel'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'right'), 'swing_start_accel'])

            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_left_step_length'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'left'), 'step_length'])
            bout_output.loc[bout_output['gait_bout_num'] == i, 'mean_right_step_length'] = np.mean(
                bout.loc[(bout['step_state'] == 'success') & (bout['foot'] == 'right'), 'step_length'])

        return bout_output

    def daily_gait(bout_times):
        bout_times['date'] = pd.to_datetime(bout_times['start_time']).dt.date
        daily_gait_dict = {
            'date': [],
            'longest_bout_length_secs': [],
            'num_bouts_over_3mins': [],
            'total_steps': []
        }

        for date, group_df in bout_times.groupby('date'):
            daily_gait_dict['date'].append(date)
            daily_gait_dict['longest_bout_length_secs'].append(group_df['bout_length_sec'].max())
            daily_gait_dict['total_steps'].append(group_df['number_steps'].sum())
            daily_gait_dict['num_bouts_over_3mins'].append(group_df.loc[group_df['bout_length_sec'] > 180].shape[0])

        daily_gait_df = pd.DataFrame(daily_gait_dict)
        daily_gait_df['day_num'] = daily_gait_df.index + 1
        daily_gait_df = daily_gait_df[
            ['day_num', 'date', 'longest_bout_length_secs', 'num_bouts_over_3mins', 'total_steps']]
        return daily_gait_df

    # # helps synchronize both bouts Not needed for WalkingBouts but could be useful for steps_df
    # if duration_sec:
    #     l_start, l_end = find_dp(left_accel_path, duration_sec, timestamp_str=start_time)
    #     r_start, r_end = find_dp(right_accel_path, duration_sec, timestamp_str=start_time)
    #     left_kwargs['start'], left_kwargs['end'] = l_start, l_end
    #     right_kwargs['start'], right_kwargs['end'] = r_start, r_end

    # left_stepdetector = StepDetection(accel_path_or_obj=left_accel_path, **left_kwargs)
    # right_stepdetector = StepDetection(accel_path_or_obj=right_accel_path,
    #                                    **right_kwargs) if left_accel_path != right_accel_path else left_stepdetector
    # self.left_step_df = left_stepdetector.export_steps()
    # self.right_step_df = right_stepdetector.export_steps()

    left_steps_df = right_steps_df if left_steps_df is None else left_steps_df
    right_steps_df = left_steps_df if right_steps_df is None else right_steps_df

    left_steps_df['step_timestamp'] = pd.to_datetime(left_steps_df['step_timestamp'])
    right_steps_df['step_timestamp'] = pd.to_datetime(right_steps_df['step_timestamp'])
    left_steps_df['foot'] = 'left'
    right_steps_df['foot'] = 'right'

    # if device.header['device_type'] == 'GNAC':
    #     left_states = left_steps_df['state_arr']'.state_arr
    #     right_states = right_stepdetector.state_arr
    #     left_steps_failed = left_stepdetector.detect_arr
    #     right_steps_failed = right_stepdetector.detect_arr

    right_freq = right_device.signal_headers[axis]['sample_rate'] if right_device is not None else left_device.signal_headers[axis]['sample_rate']
    left_freq = left_device.signal_headers[axis]['sample_rate'] if left_device is not None else right_device.signal_headers[axis]['sample_rate']

    #assert right_freq == left_freq
    if legacy_alg:
        bout_num_df = identify_bouts(left_steps_df,
                                     right_steps_df) if bout_num_df is None else bout_num_df
    else:
        left_bouts = identify_bouts_one(left_steps_df, left_freq)
        right_bouts = identify_bouts_one(right_steps_df, right_freq)
        bout_num_df = find_overlapping_times(left_bouts, right_bouts)

    bout_steps_df = export_bout_steps(bout_num_df, left_steps_df, right_steps_df)

    # #for plotting
    # self.sig_length = min(left_stepdetector.sig_length, right_stepdetector.sig_length)
    # self.left_data = left_stepdetector.data
    # self.right_data = right_stepdetector.data
    # self.start_dp = left_stepdetector.start_dp
    # self.end_dp = left_stepdetector.end_dp
    # self.timestamps = min([left_stepdetector.timestamps, right_stepdetector.timestamps], key=len)

    return bout_steps_df, bout_num_df




#############################################################################################################################
if __name__ == '__main__':
    #setup subject and filepath
    ankle_acc = nimbalwear.Device()
    #AXV6
    subj = "OND09_0011_01"
    ankle_path = fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_LAnkle.edf'
    if os.path.exists(ankle_path):
         ankle_acc.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_LAnkle.edf')
    else:
         ankle_acc.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_RAnkle.edf')
    # #GNAC
    # subj = "OND06_1027_01"
    # ankle_path = fr'W:\NiMBaLWEAR\OND06\processed\standard_device_edf\GNAC\{subj}_GNAC_LAnkle.edf'
    # if os.path.exists(ankle_path):
    #     ankle_acc.import_edf(file_path=fr'W:\NiMBaLWEAR\OND06\processed\standard_device_edf\GNAC\{subj}_GNAC_LAnkle.edf')
    # else:
    #     ankle_acc.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\sensor_edf\{subj}_GNAC_RAnkle.edf')

    # Creating step detection algorithm
    # TODO: Run step detection algorithm
    #This is what I imagine the line to look like
            # steps_df = detect_stepping(accelerometer=ankle_acc, gyroscope=None, bilateral=False)
    # TODO: Select the signals that are needed (accelerometer) for step detection
    #freq, acc_data, xz_data, timestamps, axis = get_acc_data(accelerometer=None, axis=None, orient_signal=True, low_pass=True)
    #all outcomes need to be pushed into the detect_fl_steps
   # freq, acc_data, xz_data, timestamps, axis = get_acc_data(accelerometer=ankle_acc, axis=1, orient_signal=True, low_pass=True)

    #def detect_fl_steps(device=None, data=None, start=0, end=-1,  freq=None, axis=None):
   #steps_df = detect_steps(device = ankle_acc, data = acc_data[0], start=100000, end=200000,  freq=freq, axis=axis, timestamps=timestamps, xz_data=xz_data)

    # TODO: Run steps through to find walking bouts
    #steps_df = pd.read_csv(r'W:\dev\gait\sample_steps.csv')

    #def get_walking_bouts(left_steps_df=None, right_steps_df=None, right_device=None, left_device=None, duration_sec=15, bout_num_df=None, legacy_alg=False, left_kwargs={}, right_kwargs={}):
    #bouts_steps_df, bouts_df = get_walking_bouts(left_steps_df=steps_df, left_device=ankle_acc)
    # TODO: Run walking bouts through spatiotemporal characteristics that are available for that person