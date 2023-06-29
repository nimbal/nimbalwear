from datetime import timedelta
import math
import os
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks, peak_widths, welch
import matplotlib.pyplot as plt
import pyedflib


class AccelReader:
    def __init__(self, accel_path_or_obj, label='AccelReader', axis=None, orient_signal=True, low_pass=True, start=-1,
                 end=-1, quiet=False):
        """
        AccelReader object reads accelerometer data from EDF files and loads it into the class

        Required Parameters:
        - `accel_path` (str): path to the accelerometer EDF
        - `label` (str): A label that's associated with the particular accelerometer data
        - `axis` (int): tells the EDF reader which column of the EDF to access

        Optional Parameters:
        - `start` (int): starting datapoint to splice data
        - `end` (int): ending datapoint to splice data
        - `quiet` (bool): stops printing
        """
        if not quiet:
            print("%s: Reading Accel Data..." % label)
        self.accel_path_or_obj = accel_path_or_obj
        self.freq, self.data, self.xz_data, self.timestamps, self.axis = AccelReader.get_accel_data(
            accel_path_or_obj, axis=axis, start=start, end=end)
        self.raw_data = self.data.copy()
        self.start_dp = start
        self.end_dp = end

        self.label = label
        self.quiet = quiet
        self.sig_length = len(self.data)

        if orient_signal:
            self.flip_signal()
        if low_pass:
            self.lowpass_filter()

    @staticmethod
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

    @staticmethod
    def get_accel_data_path(path, axis=None, start=-1, end=-1):
        """
        Gets the accelerometer data from EDF
        """
        accel_axes = [0, 1, 2]
        accel_file = pyedflib.EdfReader(path)

        if not (start > -1 and end > -1):
            start = 0
            end = accel_file.getNSamples()[accel_axes[0]]

        if axis is not None:
            accel_axes.remove(axis)
            data = accel_file.readSignal(axis, start=start, n=(end - start))
            xz_data = np.sqrt(
                accel_file.readSignal(accel_axes[0], start=start, n=(end - start)) ** 2 + accel_file.readSignal(
                    accel_axes[1], start=start, n=(end - start)) ** 2)
        else:
            all_data = np.array([accel_file.readSignal(ax, start=start, n=(end - start)) for ax in accel_axes])
            axis_index, test_stats = AccelReader.detect_vert(all_data[0:2])  # assumes vertical is 0 or 1
            other_axes = np.delete(np.arange(all_data.shape[0]), axis_index)
            axis = accel_axes[axis_index]
            data = all_data[axis_index]
            xz_data = np.sqrt((all_data[other_axes] ** 2).sum(axis=0))

        freq = accel_file.getSampleFrequency(axis)
        start_time = accel_file.getStartdatetime() + timedelta(0, start / freq)
        end_time = start_time + timedelta(0, accel_file.getFileDuration(
        )) - timedelta(0, accel_file.getFileDuration() - (end - start) / freq)
        timestamps = np.asarray(pd.date_range(
            start=start_time, end=end_time, periods=len(data)))

        accel_file.close()
        return freq, data, xz_data, timestamps, axis

    @staticmethod
    def get_accel_data(path_or_obj, axis=None, start=-1, end=-1):
        """

        Parameters
        ----------
        path_or_obj : TYPE
            DESCRIPTION.
        axis : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        frequency, accelerometer data, xz_data, timestamps, axis

        """
        if isinstance(path_or_obj, str):
            return AccelReader.get_accel_data_path(path_or_obj, axis, start, end)
        all_data = np.array(path_or_obj.signals)
        accel_axes = [0, 1, 2]
        if axis is not None:
            accel_axes.remove(axis)
            data = all_data[axis]
            xz_data = np.sqrt(all_data[accel_axes[0]] ** 2 + all_data[accel_axes[1]] ** 2)
        else:
            axis_index, test_stats = AccelReader.detect_vert(all_data[0:2])  # assumes vertical is 0 or 1
            other_axes = np.delete(np.arange(all_data.shape[0]), axis_index)
            axis = accel_axes[axis_index]
            data = all_data[axis_index]
            xz_data = np.sqrt((all_data[other_axes] ** 2).sum(axis=0))

        freq = path_or_obj.signal_headers[axis]['sample_rate']
        start_time = path_or_obj.header['startdate']
        file_duration = len(data) / freq
        end_time = start_time + timedelta(0, file_duration)
        timestamps = np.asarray(pd.date_range(
            start=start_time, end=end_time, periods=len(data)))

        return freq, data, xz_data, timestamps, axis

    @staticmethod
    def detect_vert(axes, method='adg'):
        """

        NOTE: To improve function when passing in axes:
                    - remove axes that are unlikely to be the vertical axis
                    - remove data points that are known to be nonwear or lying down
                    - OR only pass data from a known bout of standing

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

    @staticmethod
    def sig_init(raw_x, raw_y, raw_z, startdate, freq, **kwargs):
        """
        Initialize AccelReader object with raw signals, startdate, and freq
        raw_x, raw_y, raw_z: list
        startdate: datetime
        freq: int
        ---
        returns nwdata object
        """

        class nwdata:
            def __init__(self):
                self.header = {}
                self.signal_headers = []
                self.signals = []

        data = nwdata()
        sigs = [raw_x, raw_y, raw_z]
        data.signals.extend(sigs)
        data.signal_headers.extend([{'sample_rate': freq}] * len(sigs))
        data.header['startdate'] = startdate

        return data

    def flip_signal(self):
        """
        Finds orientation based on lowpassed signal and flips the signal
        """

        cutoff_freq = self.freq * 0.005
        sos = butter(N=1, Wn=cutoff_freq,
                            btype='low', fs=self.freq, output='sos')
        orientation = sosfiltfilt(sos, self.data)
        flip_ind = np.where(orientation < -0.25)
        self.orientation = orientation
        self.data[flip_ind] = -self.data[flip_ind]

    # 40Hz butter low pass filter
    def lowpass_filter(self, order=2, cutoff_ratio=0.17):
        """
        Applies a lowpass filter on the accelerometer data
        """
        cutoff_freq = self.freq * cutoff_ratio
        sos = butter(N=order, Wn=cutoff_freq, btype='low', fs=self.freq, output='sos')
        self.data = sosfiltfilt(sos, self.data)

    def plot(self, return_plt=False):
        """
        Plots the signal, with additional option to plot other signals that are of the same length
        """
        dp_range = np.arange(self.start_dp, self.end_dp)
        ax1 = plt.subplot(311)
        ax1.set_title('Raw Accelerometer Data')
        plt.plot(dp_range, self.raw_data, 'r-')
        plt.grid(True)

        ax2 = plt.subplot(312, sharex=ax1)
        ax2.set_title('Low Pass Filtered/Flipped Signal')
        plt.plot(dp_range, self.data, 'r-')
        plt.grid(True)

        ax3 = plt.subplot(313, sharex=ax1)
        ax3.set_title('Device Orientation')
        filtered_legend = ['Flipped Signal', 'Original Signal']
        ax3.set_yticks(np.arange(len(filtered_legend)))
        ax3.set_yticklabels(filtered_legend)
        plt.plot(dp_range, self.orientation, 'r-')
        plt.grid(True)
        plt.tight_layout()

        if not return_plt:
            plt.show()
        else:
            return plt

    @staticmethod
    def plot_sigs(start, end, signals):
        """
        Plots the signal, with additional option to plot other signals that are of the same length
        Maybe check if all signals are of the same length?
        """
        time_range = np.arange(start, end)

        fig, axs = plt.subplots(len(signals), sharex='all')
        axs = [axs] if len(signals) == 1 else axs

        for i, ax in enumerate(axs):
            ax.plot(time_range, signals[i][start:end], 'r-')

        plt.show()

class StepDetection(AccelReader):
    # increase swing threshold
    # decrease heel strike length
    push_off_threshold = 0.85
    swing_threshold = 0.5  # 0.5
    heel_strike_threshold = -5  # -5
    pushoff_time = 0.4  # 0.4 #tried 0.2 here
    swing_down_detect_time = 0.1  # 0.3
    swing_up_detect_time = 0.1  # 0.1
    swing_phase_time = swing_down_detect_time + swing_up_detect_time * 2
    heel_strike_detect_time = 0.5  # 0.8
    foot_down_time = 0.05  # 0.1 #tried 0.2 here

    def __init__(self, pushoff_df=None, label='StepDetection', **kwargs):
        """
        StepDetectioon class performs step detection through a steady state controller algorithm

        Required Parameters:
        accel_reader_obj (AccelReader): Object from AccelReader class
        pushoff_df (pandas.DataFrame): A dataframe that outlines the mean, std, and min/max for a pushoff

        Optional Parameters:
        - `quiet` (bool): stops printing
        """
        super(StepDetection, self).__init__(**kwargs)
        self.label = label
        self.pushoff_len = int(self.pushoff_time * self.freq)
        self.states = {1: 'stance', 2: 'push-off',
                       3: 'swing-up', 4: 'swing-down', 5: 'footdown'}
        self.state = self.states[1]
        self.pushoff_df = StepDetection.get_pushoff_stats(self.accel_path_or_obj,
                                                          start_end_times=[(self.start_dp, self.end_dp)],
                                                          quiet=self.quiet,
                                                          axis=self.axis) if pushoff_df is None else pushoff_df
        self.swing_peaks = []

        # set threshold values
        if {'swing_down_mean', 'swing_down_std', 'swing_up_mean', 'swing_up_std', 'heel_strike_mean',
            'heel_strike_std'}.issubset(self.pushoff_df.columns):
            self.swing_phase_time = self.pushoff_df['swing_down_mean'].iloc[0] + self.pushoff_df['swing_down_std'].iloc[
                0] + self.pushoff_df['swing_up_mean'].iloc[0] + self.pushoff_df['swing_up_std'].iloc[0]
            self.swing_phase_time = max(self.swing_phase_time, 0.1)
            self.heel_strike_detect_time = 0.5 + self.pushoff_df['swing_up_mean'].iloc[0] + 2 * \
                                           self.pushoff_df['swing_up_std'].iloc[0]
            self.heel_strike_threshold = -3 - self.pushoff_df['heel_strike_mean'].iloc[0] / (
                        2 * self.heel_strike_threshold)

        self.step_detect()

    @staticmethod
    def get_pushoff_stats(accel_path, start_end_times=[(-1, -1)], axis=None, quiet=True):
        pushoff_sig_list = []
        swingdown_times = []
        swingup_times = []
        heelstrike_values = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pushoff_df = pd.read_csv(os.path.join(dir_path, 'data', 'pushoff_OND07_left.csv'))

        for start, end in start_end_times:
            step = StepDetection(accel_path_or_obj=accel_path, label='get_pushoff_stats',
                                 axis=axis, start=start, end=end, quiet=True, pushoff_df=pushoff_df)
            pushoff_sig = StepDetection.get_pushoff_sigs(step, quiet=quiet)
            step_summary = step.export_steps()
            toe_offs = step_summary.loc[step_summary['step_state'] == 'success', 'swing_start_time']
            mid_swings = step_summary.loc[step_summary['step_state'] == 'success', 'mid_swing_time']
            heel_strikes = step_summary.loc[step_summary['step_state'] == 'success', 'heel_strike_time']
            step_indices = step_summary.loc[step_summary['step_state'] == 'success', 'step_index'] - step.start_dp

            mid_swing_indices = step_indices + (
                        step.pushoff_time + (mid_swings - toe_offs).dt.total_seconds()) * step.freq

            if len(pushoff_sig) == 0:
                print('WARNING: No steps found (start=%s, end=%s)' % (str(start), str(end)))
                continue
            pushoff_sig_list.append(pushoff_sig)
            swingdown_times.append((mid_swings - toe_offs).dt.total_seconds())
            swingup_times.append((heel_strikes - mid_swings).dt.total_seconds())
            heelstrike_values.append([np.min(step.heel_strike_detect(int(ms_ind))) for ms_ind in mid_swing_indices])

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

        # TODO: add right peaks to the detection
        # 1. find avg push_off signal

    @classmethod
    def get_pushoff_sigs(cls, step_obj, peaks=None, quiet=False):
        """
        Creates average pushoff dataframe that is used to find pushoff data
        """
        pushoff_sig_list = []
        pushoff_len = step_obj.pushoff_time * step_obj.freq

        if not peaks:
            peaks = np.array(step_obj.step_indices) + pushoff_len
        for i, peak_ind in tqdm(enumerate(peaks), desc="Generating pushoff average", total=len(peaks), disable=quiet,
                                leave=False):
            pushoff_sig = step_obj.data[int(peak_ind - pushoff_len):int(peak_ind)]
            pushoff_sig_list.append(pushoff_sig)

        return np.array(pushoff_sig_list)

    @staticmethod
    def df_from_csv(csv_file):
        """
        reads a csv file and returns a DataFrame object
        """
        return pd.read_csv(csv_file)

    def push_off_detection(self):
        """
        Detects the steps based on the pushoff_df
        """
        if not self.quiet:
            print('%s: Finding Indices for pushoff' % self.label)

        pushoff_avg = self.pushoff_df['avg']

        cc_list = StepDetection.window_correlate(self.data, pushoff_avg)

        # TODO: DISTANCE CAN BE ADJUSTED FOR THE LENGTH OF ONE STEP RIGHT NOW ASSUMPTION IS THAT A PERSON CANT TAKE 2 STEPS WITHIN 0.5s
        pushoff_ind, _ = find_peaks(cc_list, height=self.push_off_threshold, distance=max(0.2 * self.freq, 1))

        return pushoff_ind

    def step_detect(self):
        """
        Detects the steps within the accelerometer data. Based on this paper:
        https://ris.utwente.nl/ws/portalfiles/portal/6643607/00064463.pdf
        """
        pushoff_ind = self.push_off_detection()
        end_pushoff_ind = pushoff_ind + self.pushoff_len
        state_arr = np.zeros(self.data.size)
        detects = {'push_offs': len(end_pushoff_ind), 'mid_swing_peak': [], 'swing_up': [], 'swing_down': [
        ], 'heel_strike': [], 'next_i': [], 'pushoff_mean': []}
        detect_arr = np.zeros(self.data.size)

        end_i = None
        step_indices = []
        step_lengths = []

        for count, i in tqdm(enumerate(end_pushoff_ind), disable=self.quiet, total=len(end_pushoff_ind), leave=False,
                             desc='%s: Step Detection' % self.label):
            # check if next index within the previous detection
            if end_i and i - self.pushoff_len < end_i:
                detects['next_i'].append(i - 1)
                continue

            # mean/std check for pushoff, state = 1
            pushoff_mean = np.mean(self.data[i - self.pushoff_len:i])
            upper = (self.pushoff_df['avg'] + self.pushoff_df['std'])
            lower = (self.pushoff_df['avg'] - self.pushoff_df['std'])
            if not np.any((pushoff_mean < upper) & (pushoff_mean > lower)):
                detects['pushoff_mean'].append(i - 1)
                continue

            # midswing peak detection
            mid_swing_i = self.mid_swing_peak_detect(i)
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

            # heel-strike, state = 4
            accel_derivatives = self.heel_strike_detect(mid_swing_i)
            accel_threshold_list = np.where(
                accel_derivatives < self.heel_strike_threshold)[0]
            if len(accel_threshold_list) == 0:
                detects['heel_strike'].append(i - 1)
                continue
            accel_ind = accel_threshold_list[0] + mid_swing_i
            end_i = accel_ind + int(self.foot_down_time * self.freq)

            state_arr[i - self.pushoff_len:i] = 1
            state_arr[i:mid_swing_i] = 2
            state_arr[mid_swing_i:accel_ind] = 3
            state_arr[accel_ind:end_i] = 4

            step_indices.append(i - self.pushoff_len)
            step_lengths.append(end_i - (i - self.pushoff_len))

        detect_arr[detects['swing_down']] = 1
        detect_arr[detects['swing_up']] = 2
        detect_arr[detects['heel_strike']] = 3
        detect_arr[detects['next_i']] = 4
        detect_arr[detects['pushoff_mean']] = 5
        detect_arr[detects['mid_swing_peak']] = 6

        self.state_arr = state_arr
        self.step_indices = step_indices
        self.step_lengths = step_lengths
        self.detect_arr = detect_arr

        return state_arr, step_indices

    @staticmethod
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

    def mid_swing_peak_detect(self, pushoff_ind):
        swing_detect = int(self.freq * self.swing_phase_time)  # length to check for swing
        detect_window = self.data[pushoff_ind:pushoff_ind + swing_detect]
        peaks, prop = find_peaks(-detect_window,
                                        distance=max(swing_detect * 0.25, 1),
                                        prominence=0.2, wlen=swing_detect,
                                        width=[0 * self.freq, self.swing_phase_time * self.freq], rel_height=0.75)
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

        swing_down_cc = [np.corrcoef(detect_window, swing_down_sig)[0, 1]] if detect_window.shape[0] > 1 else [0]
        swing_up_cc = [np.corrcoef(swing_up_detect_window, swing_up_sig)[0, 1]] if swing_up_detect_window.shape[
                                                                                       0] > 1 else [0]

        return (swing_down_cc, swing_up_cc)

    def heel_strike_detect(self, window_ind):
        """
        Detects a heel strike based on the change in acceleration over time.
        """
        heel_detect = int(self.freq * self.heel_strike_detect_time)
        detect_window = self.data[window_ind:window_ind + heel_detect]
        accel_t_plus1 = np.append(
            detect_window[1:detect_window.size], detect_window[-1])
        accel_t_minus1 = np.insert(detect_window[:-1], 0, detect_window[0])
        accel_derivative = (accel_t_plus1 - accel_t_minus1) / (2 / self.freq)

        return accel_derivative

    def export_steps(self):
        assert len(self.detect_arr) == len(self.timestamps)
        failed_step_indices = np.where(self.detect_arr > 0)[0]
        failed_step_timestamps = self.timestamps[failed_step_indices]

        error_mapping = {1: 'swing_down', 2: 'swing_up',
                         3: 'heel_strike_too_small', 4: 'too_close_to_next_i',
                         5: 'too_far_from_pushoff_mean', 6: 'mid_swing_peak_not_detected'}
        failed_step_state = list(map(error_mapping.get, self.detect_arr[failed_step_indices]))

        step_timestamps = self.timestamps[self.step_indices]

        swing_start = np.where((self.state_arr == 1) & (np.roll(self.state_arr, -1) == 2))[0]
        mid_swing = np.where((self.state_arr == 2) & (np.roll(self.state_arr, -1) == 3))[0]
        heel_strike = np.where((self.state_arr == 3) & (np.roll(self.state_arr, -1) == 4))[0]

        pushoff_start = swing_start - int(self.pushoff_time * self.freq)
        gait_cycle_end = heel_strike + int(self.foot_down_time * self.freq)
        step_lengths = (gait_cycle_end - pushoff_start) / self.freq
        avg_speed = [np.mean(self.xz_data[i:i + int(lengths * self.freq)]) * 9.81 * lengths for i, lengths in
                     zip(self.step_indices, step_lengths)]

        assert len(self.step_indices) == len(swing_start)
        assert len(self.step_indices) == len(mid_swing)
        assert len(self.step_indices) == len(heel_strike)

        successful_steps = pd.DataFrame({
            'step_time': step_timestamps,
            'step_index': np.array(self.step_indices) + self.start_dp,
            'step_state': 'success',
            'swing_start_time': self.timestamps[swing_start],
            'mid_swing_time': self.timestamps[mid_swing],
            'heel_strike_time': self.timestamps[heel_strike],
            'swing_start_accel': self.data[swing_start],
            'mid_swing_accel': self.data[mid_swing],
            'heel_strike_accel': self.data[heel_strike],
            'step_length': step_lengths,
            'avg_speed': avg_speed
        })
        failed_steps = pd.DataFrame({
            'step_time': failed_step_timestamps,
            'step_index': np.array(failed_step_indices) + self.start_dp,
            'step_state': failed_step_state
        })
        df = pd.concat([successful_steps, failed_steps], sort=True)
        df = df.sort_values(by='step_index')
        df = df.reset_index(drop=True)

        return df

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


class WalkingBouts():
    def __init__(self, left_accel_path, right_accel_path, start_time=None, duration_sec=None, bout_num_df=None,
                 legacy_alg=False, left_kwargs={}, right_kwargs={}):
        """
        WalkingBouts class finds the bouts within two StepDetection objects.
        If you only want to analyze one ankle then make left_accel_path == right_accel_path

        Required Parameters:
        `left_stepdetector` (StepDetection): left StepDetection Class
        `right_stepdetector` (StepDetection): Right StepDetection Class
        """
        # helps synchronize both bouts
        if duration_sec:
            l_start, l_end = StepDetection.find_dp(left_accel_path, duration_sec, timestamp_str=start_time)
            r_start, r_end = StepDetection.find_dp(right_accel_path, duration_sec, timestamp_str=start_time)
            left_kwargs['start'], left_kwargs['end'] = l_start, l_end
            right_kwargs['start'], right_kwargs['end'] = r_start, r_end

        left_stepdetector = StepDetection(accel_path_or_obj=left_accel_path, **left_kwargs)
        right_stepdetector = StepDetection(accel_path_or_obj=right_accel_path,
                                           **right_kwargs) if left_accel_path != right_accel_path else left_stepdetector
        self.left_step_df = left_stepdetector.export_steps()
        self.right_step_df = right_stepdetector.export_steps()
        self.left_step_df['step_time'] = pd.to_datetime(self.left_step_df['step_time'])
        self.right_step_df['step_time'] = pd.to_datetime(self.right_step_df['step_time'])
        self.left_step_df['foot'] = 'left'
        self.right_step_df['foot'] = 'right'

        self.left_states = left_stepdetector.state_arr
        self.right_states = right_stepdetector.state_arr
        self.left_steps_failed = left_stepdetector.detect_arr
        self.right_steps_failed = right_stepdetector.detect_arr
        self.freq = left_stepdetector.freq  # TODO: check if frequencies are the same

        assert left_stepdetector.freq == right_stepdetector.freq
        if legacy_alg:
            self.bout_num_df = WalkingBouts.identify_bouts(left_stepdetector,
                                                           right_stepdetector) if bout_num_df is None else bout_num_df
        else:
            left_bouts = WalkingBouts.identify_bouts_one(left_stepdetector)
            right_bouts = WalkingBouts.identify_bouts_one(right_stepdetector)
            self.bout_num_df = WalkingBouts.find_overlapping_times(left_bouts, right_bouts)

        self.sig_length = min(left_stepdetector.sig_length, right_stepdetector.sig_length)
        self.left_data = left_stepdetector.data
        self.right_data = right_stepdetector.data
        self.start_dp = left_stepdetector.start_dp
        self.end_dp = left_stepdetector.end_dp
        self.timestamps = min([left_stepdetector.timestamps, right_stepdetector.timestamps], key=len)

    @staticmethod
    def identify_bouts_one(step_detector):
        freq = step_detector.freq
        steps = step_detector.step_indices
        timestamps = step_detector.timestamps[steps]
        step_lengths = step_detector.step_lengths

        steps_df = pd.DataFrame({'step_index': steps, 'timestamp': timestamps, 'step_length': step_lengths})
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
                    start_ind = start_step['step_index']
                    end_ind = curr_step['step_index'] + curr_step['step_length']
                    bout_dict['start'].append(start_ind)
                    bout_dict['end'].append(end_ind)
                    bout_dict['number_steps'].append(step_count)
                    bout_dict['start_time'].append(start_step['timestamp'])
                    bout_dict['end_time'].append(
                        curr_step['timestamp'] + pd.Timedelta(curr_step['step_length'] / freq, unit='sec'))

                # resets state and creates new bout
                step_count = 1
                next_curr_steps = steps_df.loc[steps_df['timestamp'] > curr_step['timestamp']]
                curr_step = next_curr_steps.iloc[0] if not next_curr_steps.empty else None
                start_step = curr_step
                next_steps = None

        bout_num_df = pd.DataFrame(bout_dict)
        return bout_num_df

    @staticmethod
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
            return WalkingBouts.find_overlapping_times(df, pd.DataFrame())

    @staticmethod
    def identify_bouts(left_stepdetector, right_stepdetector):
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

    def export_split_bouts(self, output_dir='figures'):
        """
        Takes in output dir and outputs images of the bouts into the output dir as .png images
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, row in self.bout_num_df.iterrows():
            fig = self.plot_single_bout(i, True)
            fig.savefig(os.path.join(output_dir, 'plot%d.png' % (i)))
            fig.clf()

    def plot_single_bout(self, bout_num, return_plt=False):
        """
        Shows single bout or start and end time range.
        Input is either the bout_num
        """
        row = self.bout_num_df.iloc[bout_num]
        pad_start = max(row['start'] - self.freq, 0)
        pad_end = min(row['end'] + self.freq, self.sig_length)
        fig = self.plot(pad_start, pad_end, return_plt)
        if not return_plt:
            fig.show()
        else:
            return fig

    def export_steps(self):
        bout_steps = []
        for i, row in self.bout_num_df.iterrows():
            start = row['start_time'] - pd.Timedelta(1, unit='sec')
            end = row['end_time'] + pd.Timedelta(1, unit='sec')

            left_bout_step_df = self.left_step_df.loc[
                (self.left_step_df['step_time'] > start) & (self.left_step_df['step_time'] < end)]
            right_bout_step_df = self.right_step_df.loc[
                (self.right_step_df['step_time'] > start) & (self.right_step_df['step_time'] < end)]

            bout_step_df = pd.concat([left_bout_step_df, right_bout_step_df])
            bout_step_df['gait_bout_num'] = i + 1
            bout_steps.append(bout_step_df)

        if len(bout_steps) == 0:
            return pd.DataFrame()

        bout_step_summary = pd.concat(bout_steps)
        bout_step_summary.sort_values(by=['gait_bout_num', 'foot', 'step_time'])
        bout_step_summary = bout_step_summary.reset_index()
        bout_step_summary['step_num'] = bout_step_summary.index + 1

        return bout_step_summary

    def plot_accel(self, start=-1, end=-1, return_plt=False, margin=3):
        if not (start > -1 and end > -1):
            start = 0
            end = self.sig_length

        margin_dp = margin * self.freq
        duration = end - start
        start = start - margin_dp if start - margin_dp > 0 else 0
        end = end + margin_dp if end + margin_dp < self.sig_length else self.sig_length - 1

        if start == 0 or end == self.sig_length:
            margin = 0

        start_time = start / self.freq
        time_range = np.linspace(-margin, duration / self.freq + margin, num=end - start)

        bouts = np.zeros(self.sig_length)
        for i, row in self.bout_num_df.iterrows():
            bouts[row['start']:row['end']] = 5
        states_legend = ['stance', 'pushoff', 'swing down', 'swing up', 'heel strike']

        fig, ax = plt.subplots(1, 1, sharex='all', figsize=(15, 8))

        ax.set_title('Left/Right Accelerometer Data')
        ax.plot(time_range, self.left_data[start:end], 'r-', label='Left Accel')
        ax.plot(time_range, self.right_data[start:end], 'b-', label='Right Accel')
        # ax.axvline(time_range[0] + margin)
        # ax.axvline(time_range[-1] - margin)
        ax.set_xticks(np.arange(min(time_range), max(time_range) + 1, 1), minor=True)
        ax.legend(loc='upper left')
        ax.grid(which='both')

        fig.tight_layout()
        if not return_plt:
            fig.show()
        else:
            return fig

    def plot(self, start=-1, end=-1, return_plt=False, include_bouts=True, margin=3):
        """
        Plots the overlayed accelerometer data, left and right states in accel data, and bouts detected
        ***margin is in seconds
        """
        if not (start > -1 and end > -1):
            start = 0
            end = self.sig_length

        margin_dp = margin * self.freq
        duration = end - start
        start = start - margin_dp if start - margin_dp > 0 else 0
        end = end + margin_dp if end + margin_dp < self.sig_length else self.sig_length - 1

        if start == 0 or end == self.sig_length:
            margin = 0

        start_time = start / self.freq
        time_range = np.linspace(-margin, duration / self.freq + margin, num=end - start)

        bouts = np.zeros(self.sig_length)
        for i, row in self.bout_num_df.iterrows():
            bouts[int(row['start']):int(row['end'])] = 5
        states_legend = ['stance', 'pushoff', 'swing down', 'swing up', 'heel strike']

        fig, axs = plt.subplots(3, 1, sharex='all', figsize=(15, 8))

        axs[0].set_title('Left/Right Accelerometer Data')
        axs[0].plot(time_range, self.left_data[start:end], 'r-', label='Left Accel')
        axs[0].plot(time_range, self.right_data[start:end], 'b-', label='Right Accel')
        axs[0].axvline(time_range[0] + margin)
        axs[0].axvline(time_range[-1] - margin)
        axs[0].set_xticks(np.arange(min(time_range), max(time_range) + 1, 1), minor=True)
        axs[0].grid(which='both')

        axs[1].plot(time_range, self.left_states[start:end], "r-", label='Left Accel')
        axs[1].plot(time_range, self.right_states[start:end], "b-", label='Right Accel')
        axs[1].axvline(time_range[0] + margin)
        axs[1].axvline(time_range[-1] - margin)
        axs[1].set_xticks(np.arange(min(time_range), max(time_range) + 1, 1), minor=True)
        axs[1].fill_between(time_range, 0, bouts[start:end], alpha=0.5)
        axs[1].set_title('States of Steps in Accelerometer Data')
        axs[1].set_yticks(np.arange(len(states_legend)))
        axs[1].set_yticklabels(states_legend)
        axs[1].legend(loc='upper left')
        axs[1].grid(which='both')

        if include_bouts:
            axs[2].plot(time_range, bouts[start:end], "g-")
            axs[2].set_title('Bouts Detected')
            axs[2].set_xlabel('Time (s)')
            axs[2].grid(which='both')
        else:
            filtered_legend = ['', 'swing down', 'swing up', 'heel strike',
                               'next pushoff too close', 'pushoff mean too far', 'mid_swing_peak']
            axs[2].plot(time_range, self.left_steps_failed[start:end], "ro")
            axs[2].plot(time_range, self.right_steps_failed[start:end], "bo")
            axs[2].set_xticks(np.arange(min(time_range), max(time_range) + 1, 1), minor=True)
            axs[2].set_yticks(np.arange(len(filtered_legend)))
            axs[2].set_yticklabels(filtered_legend)
            axs[2].set_title('Failed Steps')
            axs[2].set_xlabel('Time (s)')
            axs[2].grid(which='both')

        fig.tight_layout()
        axs[2].axvline(time_range[0] + margin)
        axs[2].axvline(time_range[-1] - margin)

        if not return_plt:
            fig.show()
        else:
            return fig

    def export_bouts(self, name='UNKNOWN', verbose=False):
        summary = pd.DataFrame({
            'name': name,
            'gait_bout_num': self.bout_num_df.index + 1,
            'start_time': self.bout_num_df['start_time'],
            'end_time': self.bout_num_df['end_time'],
            'start_dp': self.bout_num_df['start'],
            'end_dp': self.bout_num_df['end'],
            'bout_length_dp': self.bout_num_df['end'] - self.bout_num_df['start'],
            'bout_length_sec': [(row['end_time'] - row['start_time']).total_seconds() for i, row in
                                self.bout_num_df.iterrows()],
            'step_count': self.bout_num_df['number_steps']
        })
        if self.left_states.shape == self.right_states.shape:
            states = self.left_states + self.right_states
            summary['gait_time_sec'] = [
                len(np.where(states[int(row['start_dp']):int(row['end_dp'])] > 0)[0]) / self.freq for i, row in
                summary.iterrows()]

        if verbose:
            summary = self.verbose_bout_output(summary)
        return summary

    def verbose_bout_output(self, bout_output):
        bout_step_summary = self.export_steps()

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

    @staticmethod
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

def get_signals(data, index_dict, gyro_axis='z', acc_axis='x', start_index=0, stop_index=None):
    # import vertical acceleration for plotting
    # import ML/sagittal gyro for step detection

    gyro = np.array(data[index_dict[f"gyro_{gyro_axis}"]][start_index:stop_index])
    acc = np.array(data[index_dict[f"accel_{acc_axis}"]][start_index:stop_index])

    return gyro, acc


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
    sos = butter(N=order, Wn=fc, btype='low', output='sos', fs=fs)
    filtered_data = sosfiltfilt(sos, data)
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


def get_gait_bouts(data, sample_freq,  timestamps, break_sec=2, bout_steps=3, start_ind=0, end_ind=None,):

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

    # setting up bout counting - this will be for
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
    step_events_df['Peak_times'] = timestamps[step_events_df['Step_index']]

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

    return step_events_df, gait_bouts_df, peak_heights

    # bouts.to_csv(r'Z:\OBI\ONDRI@Home\Participant Summary Data - Feedback\HANDDS Feedback Forms\Summary Dataframes\Step counting checks\bouts_dataframe.csv')


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
