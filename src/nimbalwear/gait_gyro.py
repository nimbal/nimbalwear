from datetime import timedelta
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


def fraccaro_gyro_steps(gyro, freq, start_time=None):
    """
    Detects the steps within the gyroscope data. Based on this paper:
    Fraccaro, P., Coyle, L., Doyle, J., & O'Sullivan, D. (2014). Real-world gyroscope-based gait event detection and
    gait feature extraction.
    """

    lf_data = bw_filter(gyro, freq, 3, 5)

    th1 = find_adaptive_thresh(lf_data, freq)

    idx_peaks, peak_hghts = find_peaks(x=gyro, height=th1, distance=(.8 * freq))

    step_count = range(1, len(idx_peaks) + 1)
    steps = pd.DataFrame({'step_num': step_count, 'step_idx': idx_peaks})

    if start_time is not None:

        step_times = pd.Series([start_time + timedelta(seconds=(i / freq)) for i in steps['step_idx']])
        steps.insert(loc=1, column='step_time', value=step_times)

    return steps
