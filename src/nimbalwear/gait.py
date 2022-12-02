from datetime import timedelta
import math
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, sosfilt, find_peaks, peak_widths, welch
import nimbalwear

def detect_steps(data = None, loc=None, ra_data=None, la_data = None, data_type='acc',  start=0, end=-1,
                 orient_signal=True, low_pass=True):
    '''
    Parameters
    ---
    ra_data -> right ankle data
    la_data -> left ankle data
    data_type ->
    start -> where do you want to start your step detection
    end -> where do you want step detection to end
    ---
    Returns
    ---
    steps_df -> dataframe with detected steps
    '''


    #define functions

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

            return vert_idx, test_stats

    def lowpass_filter(acc_data, freq, order=2, cutoff_ratio=0.17):
            """
            Applies a lowpass filter on the accelerometer data
            """
            cutoff_freq = freq * cutoff_ratio
            sos = butter(N=order, Wn=cutoff_freq,
                         btype='low', fs=freq, output='sos')
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

    def detect_steps_ssc(data=None,  start_dp=start, end_dp=end, axis=None,
                         freq=fs, pushoff_df=True, timestamps=None, xz_data=None):
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
        def export_steps(detect_arr=None, state_arr=None, timestamps=None, step_indices=None, start_dp=None,
                         pushoff_time=None, foot_down_time=None, success=True):
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
            #avg_speed = [np.mean(xz_data[i:i + int(lengths * freq)]) * 9.81 * lengths for i, lengths in
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
                'step_duration': step_durations
                #'avg_speed': avg_speed
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

        def push_off_detection(data=None, pushoff_df=None, push_off_threshold=None, freq=None):
            """
            Detects the steps based on the pushoff_df, uses window correlate and cc threshold  to accept/reject pushoffs
            """
            pushoff_avg = pushoff_df['avg']

            cc_list = window_correlate(data, pushoff_avg)

            # TODO: Postponed -- DISTANCE CAN BE ADJUSTED FOR THE LENGTH OF ONE STEP RIGHT NOW ASSUMPTION IS THAT A PERSON CANT TAKE 2 STEPS WITHIN 0.5s
            pushoff_ind, _ = find_peaks(cc_list, height=push_off_threshold, distance=max(0.2 * freq, 1))

            return pushoff_ind

        def mid_swing_peak_detect(data=None, pushoff_ind=None, swing_phase_time=None):
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

        def swing_detect(self, pushoff_ind, mid_swing_ind):
            """
            Detects swings (either up or down) given a starting index (window_ind).
            Swing duration is preset - currently unused and mid_swing_peak_detect is used in place of this function
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
            Detects a heel strike based on the change in acceleration over time (first derivative).
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
        states = {1: 'stance', 2: 'push-off',
                  3: 'swing-up', 4: 'swing-down', 5: 'footdown'}
        state = states[1]

        #defining step pushoff thresholds
        if pushoff_df == True: #importing static pushoff_df
            dir_path = os.path.dirname(os.path.realpath(__file__))
            pushoff_df = pd.read_csv(os.path.join(dir_path, 'data', 'pushoff_OND07_left.csv'))
        elif pushoff_df == False:
            print('No pushoff_df available, to fix define pushoff_df')
            #pushoff_df = get_pushoff_stats(data, start_end_times=[(start_dp, end_dp)], axis=axis)

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

    def detect_steps_gyro(start_dp=None, end_dp=None):
        '''
        Detects the steps within the gyroscope data. Based on this paper:
        Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014). Real-world gyroscope-based gait event detection and gait feature extraction.
        '''
        #define functions
        def bw_filter(data, fs, fc,  order):
            """
            Filter (filtfilt) data with dual pass lowpass butterworth filter
            """
            b, a = butter(N=order, Wn=fc, btype='low', output='ba', fs=fs)
            filtered_data = filtfilt(b, a, data)

            return filtered_data

        def find_adaptive_thresh(data, fs):
            '''
            Finds adaptive threshold on preprocessed data  with minimum 40 threshold

            B.R. Greene, et al., ”Adaptive estimation of temporal gait parameters using body-worn gyroscopes,”
            Proc. IEEE Eng. Med. Bio. Soc. (EMBC 2011), pp. 1296-1299, 2010 and outlined in Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014)
            '''
            # calculate derivative of signal
            data_2d = np.diff(data)/ (1 / fs)

            # adaptive threshold from 10 max peaks in signal derivative
            thresh = np.mean(data[np.argpartition(data_2d, 10)[:10]])*0.2
            if thresh > 40:
                pass
            else:
                thresh = 40

            return thresh

        def remove_single_step_bouts(df, steps_length=2):
            '''
            Step events are imported and bouts that have less than"steps_length" amount are removed from bouts_df
            '''
            sum_df = df.groupby(['Bout_number']).count()
            sum_df.columns = ['Step_number', 'Step_index', 'Peak_times']

            sum_df.drop(sum_df[sum_df.Step_number < steps_length].index, inplace=True)
            bout_index = sum_df.index

            df= df[df.Bout_number.isin(bout_index)]
            df.reset_index(inplace=True, drop=True)
            df = renumber_bouts(df)

            return df

        def renumber_bouts(df):
            '''
            Renumbering the gait bouts for bouts_df after single step bouts are removed
            '''
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
            '''
            imported steps
            '''
            bout_list = df['Bout_number'].unique()
            bout_df = pd.DataFrame(columns=['Bout_number', 'Step_count', 'Start_time', 'End_time', 'Start_idx', 'End_idx'])
            for count, val in enumerate(bout_list):
                temp = df[df['Bout_number'] == bout_list[count]]
                step_count = len(temp)
                start_time = np.min(temp['Peak_times'])
                end_time = np.max(temp['Peak_times'])
                start_ind = np.min(temp['Step_index'])
                end_ind = np.max(temp['Step_index'])
                # cadence = step_count/((end_ind-start_ind)*(1/fs)/60)
                data = pd.DataFrame([[count + 1,  step_count, start_time, end_time, start_ind, end_ind]], columns=bout_df.columns)    # , "Cadence":cadence}
                bout_df = pd.concat([bout_df, data], ignore_index=True)

            return bout_df

        def gyro_bout_analysis(data, gait_bouts_df=None, sample_freq=None, min_swing_t=0.250, max_swing_t=0.800):
            '''
            Iterate through gait bouts to find the footfall data
            '''
            # create index values for min and maximum swing time
            min_swing_idx = sample_freq * min_swing_t
            max_swing_idx = sample_freq * max_swing_t

            peaks = []
            gyro_mean = []
            thresholds = []
            mnf = []
            mdf = []

            initial_contacts = []
            terminal_contacts = []

            for row in gait_bouts_df.itertuples():
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
                peaks.append(peak_idx)

                ics = []
                tcs = []

                for i in range(len(peak_idx)):
                    window_len = math.floor(0.4 * sample_freq)

                    # adaptation to the Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014) description of this algorithm
                    # final contacts (toe-down or flat foot) doesn't always pick off the middle peak, specifically with spikes at initail contact
                    # fc, _ = sp.find_peaks(data[int(row[5]):int(row[6])][peak_idx[i]:peak_idx[i+1]], distance=len(data[int(row[5]):int(row[6])])*0.5)
                    # print(fc)
                    # as per modification above - window_len is described here as half the distance between the two peaks

                    tc = np.argmin(data[int(peak_idx[i] - window_len):peak_idx[i]]) + peak_idx[i] - window_len
                    ic = np.argmin(data[peak_idx[i]:int(peak_idx[i] + window_len)]) + peak_idx[i]

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
                temp_mdf = freqs[np.argmin(np.abs(np.cumsum(psd) - (0.5 * sum(psd))))]
                mnf.append(temp_mnf)
                mdf.append(temp_mdf)

            gait_bouts_df['Gryo_z_mean'] = gyro_mean
            gait_bouts_df['Threshold'] = thresholds
            gait_bouts_df['MS_peaks'] = peaks
            gait_bouts_df['Initial_contacts'] = initial_contacts
            gait_bouts_df['Terminal_contacts'] = terminal_contacts
            gait_bouts_df['Mean power freq'] = mnf
            gait_bouts_df['Median power freq'] = mdf

            # need to correct start idx and end idx (plus timestamps)

            return gait_bouts_df

        def get_gait_bouts(data, sample_freq,  timestamps, break_sec=2, bout_steps=3, start_ind=0, end_ind=None):

            # crop data
            data = data[start_ind:end_ind]

            # low pass filter at 3 hz
            # 1: LP filter data at 3 Hz
            lf_data = bw_filter(data=data, fs=sample_freq, fc=3, order=5)

            # 2: Calculate adaptive threshold
            th1 = find_adaptive_thresh(data=lf_data, fs=sample_freq)

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

            step_count = range(1, len(idx_peaks) + 1)
            step_events_df = pd.DataFrame({'Step': step_count, 'Step_index': idx_peaks, 'Bout_number': bouts})

            # 4: Ensure left and right occurrence of gait events
            # single IMU only - no need for this step atm

            #get step timestamps
            step_events_df['Step_timestamp'] = timestamps[step_events_df['Step_index']]
            gait_bouts_df = remove_single_step_bouts(df=step_events_df, steps_length=bout_steps)
            gait_bouts_df = get_bouts_data(df=gait_bouts_df)
            # renumber the bouts in steps_df
            step_events_df['Bout_number'] = 0
            for i in range(len(gait_bouts_df)):
                bool = (step_events_df.Step_index >= gait_bouts_df.Start_idx[i]) & (
                            step_events_df.Step_index <= gait_bouts_df.End_idx[i])
                idx = step_events_df.index[bool]
                step_events_df.Bout_number.iloc[idx] = gait_bouts_df.Bout_number[i]

            step_events_df.columns = ['step_number', 'step_index','bout_number', 'step_timestamp']

            return step_events_df, peak_heights

        #define parameters
        all_data = np.array(device.signals)
        ## get signal labels
        index_dict = {"gyro_x": device.get_signal_index('Gyroscope x'),
                      "gyro_y": device.get_signal_index('Gyroscope y'),
                      "gyro_z": device.get_signal_index('Gyroscope z'),
                      "accel_x": device.get_signal_index('Accelerometer x'),
                      "accel_y": device.get_signal_index('Accelerometer y'),
                      "accel_z": device.get_signal_index('Accelerometer z')}
        ##get signal frequnecies needed for step detection
        gyro_freq = device.signal_headers[index_dict['gyro_x']]['sample_rate']

        gyro_data = device.signals[index_dict['gyro_z']]

        data_start_time = device.header["start_datetime"] #if start is None else start
        data_len = len(gyro_data)
        file_duration = data_len / gyro_freq
        end_time = data_start_time + timedelta(0, file_duration)
        timestamps = np.asarray(pd.date_range(start=data_start_time, end=end_time, periods=len(gyro_data)))

        steps_df, peak_heights = get_gait_bouts(data=gyro_data, sample_freq=gyro_freq, timestamps=timestamps, break_sec=2, bout_steps=3,
                                                      start_ind=0, end_ind=None)

        return steps_df

    if orient_signal:
        acc_data = flip_signal(acc_data, freq)

    if low_pass:
        acc_data, _ = lowpass_filter(acc_data, freq)

    return steps_df, bouts_df

# Walkingbouts declassed
def get_walking_bouts(steps_df=None, duration_sec=15, bout_num_df=None,
                      legacy_alg=False, left_kwargs={}, right_kwargs={}):
    """

    """

    def find_dp(path, duration_sec, timestamp_str=None, axis=1):
        """
        Gets start and end time based on a timestamp and duration_sec (# data points)
        """
        time_delta = pd.to_timedelta(freq, start_datetime, unit='s')
        start = 0
        if timestamp_str:
            start = int((pd.to_datetime(timestamp_str) -
                         start_datetime) / time_delta)
        end = int(start + pd.to_timedelta(duration_sec, unit='s') / time_delta)

        return start, end

    def identify_bouts_one(steps_df, freq):

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
            termination_bout_window = pd.Timedelta(15, unit='sec') if next_steps is None else pd.Timedelta(10,
                                                                                                           unit='sec')
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

        bout_num_df = pd.DataFrame(bout_dict)
        return bout_num_df

    def find_overlapping_times(left_bouts, right_bouts):
        # merge based on step index
        export_dict = {'start': [], 'end': [], 'step_count': [], 'start_time': [], 'end_time': []}
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
                    export_dict['step_count'].append(row['step_count'])
                    export_dict['start_time'].append(row['start_time'])
                    export_dict['end_time'].append(row['end_time'])
            else:
                export_dict['start'].append(intersect['start'].min())
                export_dict['end'].append(intersect['end'].max())
                export_dict['step_count'].append(intersect['step_count'].sum())
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

    #assert right_freq == left_freq
    if legacy_alg:
        bout_num_df = identify_bouts(left_steps_df,
                                     right_steps_df) if bout_num_df is None else bout_num_df
    else:
        left_bouts = identify_bouts_one(left_steps_df, left_freq)
        right_bouts = identify_bouts_one(right_steps_df, right_freq)
        bout_num_df = find_overlapping_times(left_bouts, right_bouts)

    bout_steps_df = export_bout_steps(bout_num_df, left_steps_df, right_steps_df)
    bouts_stats = gait_stats(bout_num_df, type='daily', single_leg=False)

    return bout_steps_df, bout_num_df, bouts_stats

########################################################################################################################
if __name__ == '__main__':
    #setup subject and filepath
    ankle = nimbalwear.Device()

    # #AXV6
    # subj = "OND09_0011_01"
    # ankle_path = fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_LAnkle.edf'
    # if os.path.exists(ankle_path):
    #      ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_LAnkle.edf')
    # else:
    #      ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\device_edf_cropped\{subj}_AXV6_RAnkle.edf')
    #GNAC
    subj = "OND06_1027_01"
    ankle_path = fr'W:\NiMBaLWEAR\OND06\processed\standard_device_edf\GNAC\{subj}_GNAC_LAnkle.edf'
    if os.path.exists(ankle_path):
        ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND06\processed\standard_device_edf\GNAC\{subj}_GNAC_LAnkle.edf')
    else:
        ankle.import_edf(file_path=fr'W:\NiMBaLWEAR\OND09\wearables\sensor_edf\{subj}_GNAC_RAnkle.edf')

    acc = ankle.signals()


    #
    # #Input for detect steps is "Device" obj
    # steps_df = detect_steps(device = ankle, bilateral_wear = False, start=100000, end=200000)
    # steps_df.to_csv(r'W:\dev\gait\acc_steps_df.csv')
    #
    # #def get_walking_bouts(left_steps_df=None, right_steps_df=None, right_device=None, left_device=None, duration_sec=15, bout_num_df=None, legacy_alg=False, left_kwargs={}, right_kwargs={}):
    # bouts_steps_df, bouts_df, bouts_stats = get_walking_bouts(left_steps_df=steps_df, left_device=ankle)
    # bouts_steps_df.to_csv(r'W:\dev\gait\acc_sample_bouts_steps_df.csv')
    # bouts_df.to_csv(r'W:\dev\gait\acc_sample_bouts_num_df.csv')
    # bouts_stats.to_csv(r'W:\dev\gait\acc_sample_bouts_stats.csv')
