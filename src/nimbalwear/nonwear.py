# Adam Vert
# December 5, 2021

# ======================================== IMPORTS ========================================
import numpy as np
import pandas as pd

from vertdetach import vertdetach


# ======================================== FUNCTIONS========================================
"""
This stores the three algorithms
"""


def vanhees_nonwear(x_values, y_values, z_values, non_wear_window=60.0, window_step_size=15, std_thresh_mg=13.0,
                    value_range_thresh_mg=50.0, num_axes_required=2, freq=75.0, quiet=False):
    """
    Calculated non-wear predictions based on the GGIR algorithm created by Vanhees
    https://cran.r-project.org/web/packages/GGIR/vignettes/GGIR.html#non-wear-detection
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061691

    Args:
        x_values: numpy array of the accelerometer x values
        y_values: numpy array of the accelerometer y values
        z_values: numpy array of the accelerometer z values
        non_wear_window: window size in minutes
        window_step_size: the distance in minutes that the window will step between loops
        std_thresh_mg: the value which the std of an axis in the window must be below
        value_range_thresh_mg: the value which the value range of an axis in the window must be below
        num_axes_required: the number of axes that must be below the std threshold to be considered NW
        freq: frequency of accelerometer in hz
        quiet: Whether or not to quiet print statements

    Returns:
        A numpy array with the length of the accelerometer data marked as either wear time (0) or non-wear time (1)

    """
    if not quiet:
        print("Starting Vanhees Calculation...")

    # Change from minutes in input to data points
    non_wear_window = int(non_wear_window * freq * 60)
    window_step_size = int(window_step_size * freq * 60)

    # Make thresholds from mg to g
    std_thresh_g = std_thresh_mg / 1000
    value_range_thresh_g = value_range_thresh_mg / 1000

    # Create array with all the raw data in it with their respective timestamps
    data = np.array([x_values, y_values, z_values])

    # Initially assuming all wear time, create a vector of wear (1) and input non-wear (0) later
    non_wear_vector = np.zeros(data.shape[1], dtype=bool)

    # Loop over data
    for n in range(0, data.shape[1], window_step_size):
        # Define Start and End points of window
        start = n
        end = start + non_wear_window

        # Grab data in window
        windowed_vector = data[:, start:end]

        # Remove final window of collection to maintain uniform window sizes
        if windowed_vector.shape[1] < non_wear_window:
            break

        # Calculate std
        window_std = windowed_vector.astype(float).std(axis=1)

        # Check how many axes are below std threshold
        std_axes_count = (window_std < std_thresh_g).sum()

        # Calculate value range
        window_value_range = np.ptp(windowed_vector, axis=1)

        # Check how many axes are below value range threshold
        value_range_axes_count = (window_value_range < value_range_thresh_g).sum()

        if (value_range_axes_count >= num_axes_required) or (std_axes_count >= num_axes_required):
            non_wear_vector[start:end] = True

    # Border Criteria
    df = pd.DataFrame({'NW Vector': non_wear_vector,
                       'idx': np.arange(len(non_wear_vector)),
                       "Unique Period Key": (pd.Series(non_wear_vector).diff(1) != 0).astype('int').cumsum(),
                       'Duration': np.ones(len(non_wear_vector))})
    period_durations = df.groupby('Unique Period Key').sum() / (freq * 60)
    period_durations['Wear'] = [False if val == 0 else True for val in period_durations['NW Vector']]
    period_durations['Adjacent Sum'] = period_durations['Duration'].shift(1, fill_value=0) + period_durations[
        'Duration'].shift(-1, fill_value=0)
    period_durations['Period Start'] = df.groupby('Unique Period Key').min()['idx']
    period_durations['Period End'] = df.groupby('Unique Period Key').max()['idx']+1
    for index, row in period_durations.iterrows():
        if not row['Wear']:
            if row['Duration'] <= 180:
                if row['Duration'] / row['Adjacent Sum'] < 0.8:
                    non_wear_vector[row['Period Start']:row['Period End']] = True
            elif row['Duration'] <= 360:
                if row['Duration'] / row['Adjacent Sum'] < 0.3:
                    non_wear_vector[row['Period Start']:row['Period End']] = True
    if not quiet:
        print("Finished Vanhees Calculation.")
    return non_wear_vector


def zhou_nonwear(x_values, y_values, z_values, temperature_values, accelerometer_frequency=75.0,
                 temperature_frequency=0.25, non_wear_window=1, window_step_size=1 / 15, t0=26,
                 std_thresh_mg=13.0, num_axes_required=3, quiet=False):
    """
    Calculated non-wear results based on the algorithm created by Shang-Ming Zhou
    https://bmjopen.bmj.com/content/5/5/e007447

    Args:
        x_values: numpy array of the accelerometer x values
        y_values: numpy array of the accelerometer y values
        z_values: numpy array of the accelerometer z values
        temperature_values: numpy array of the temperature values
        accelerometer_frequency: frequency of accelerometer in hz
        temperature_frequency: frequency of temperature in hz
        non_wear_window: window size in minutes
        window_step_size: the distance in minutes that the window will step between loops
        t0:
        std_thresh_mg: the value which the std of an axis in the window must be below
        num_axes_required: the number of axes that must be below the std threshold to be considered NW
        quiet: Whether or not to quiet print statements

    Returns:
        A numpy array with the length of the accelerometer data marked as either wear time (0) or non-wear time (1)

    """
    if not quiet:
        print("Starting Zhou Calculation...")

    # Change from minutes in input to data points
    non_wear_window_accelerometer = int(non_wear_window * accelerometer_frequency * 60)
    non_wear_window_temperature = int(non_wear_window * temperature_frequency * 60)
    window_overlap_accelerometer = int(window_step_size * accelerometer_frequency * 60)
    window_overlap_temperature = int(window_step_size * temperature_frequency * 60)

    # Calculate the number of overlaps in a window
    num_overlaps_in_window = non_wear_window / window_step_size

    # Make thresholds from mg to g
    std_thresh_g = std_thresh_mg / 1000

    # Create array with all the raw data
    accelerometer_data = np.array([x_values, y_values, z_values])

    # Initially assuming all wear time, create a vector with length of accelerometer data with wear (1)
    # and input non-wear (0) later
    non_wear_vector = np.ones(accelerometer_data.shape[1], dtype=np.uint8)

    # Create a blank list to put the windowed temperature average values
    windowed_temperature_list = []

    # Create a blank list to put the results for whether the windowed accelerometer std value below the threshold (1)
    # or above (0)
    windowed_accelerometer_list = []

    # Solve for temperature values for each window, returning the average temperature value for the window
    for n in range(0, len(temperature_values), window_overlap_temperature):

        # Define start and end points for the window
        start = n
        end = start + non_wear_window_temperature

        # Create vector of windowed temperature_values
        windowed_temperature_values = temperature_values[start:end]

        # Remove final window to maintain uniform size
        if len(windowed_temperature_values) < non_wear_window_temperature:
            break

        # Get average temperature in the window
        window_average_temperature = np.average(windowed_temperature_values)

        # Append windowed average temperature to the previously created list
        windowed_temperature_list.append(window_average_temperature)

    # Windowed std
    for n in range(0, accelerometer_data.shape[1], window_overlap_accelerometer):
        below_thresh = 0

        # Define Start and End points of window
        start = n
        end = start + non_wear_window_accelerometer

        # Grab data in window
        windowed_vector = accelerometer_data[:, start:end]

        # Remove final window of collection to maintain uniform window sizes
        if windowed_vector.shape[1] < non_wear_window_accelerometer:
            break

        # Calculate std
        window_std = windowed_vector.astype(float).std(axis=1)

        # Check how many axes are below std threshold, if below the requirement mark as  NW
        std_axes_count = (window_std < std_thresh_g).sum()
        if std_axes_count >= num_axes_required:
            below_thresh = 1

        windowed_accelerometer_list.append(below_thresh)

    # Check to see if the temperature and accelerometer lists are the same length, if they are +-1 then make them
    # the same length
    if len(windowed_temperature_list) != len(windowed_accelerometer_list):
        if len(windowed_temperature_list) - len(windowed_accelerometer_list) == 1:
            windowed_temperature_list.pop()
        elif len(windowed_accelerometer_list) - len(windowed_temperature_list) == 1:
            windowed_accelerometer_list.pop()
        else:
            raise Exception("Somehow the temperature and accelerometer lists have very different sizes.")

    # Create array of windowed results
    windowed_results = np.array([windowed_temperature_list, windowed_accelerometer_list])

    for n in range(0, windowed_results.shape[1]):

        # Define start and end times of window
        start = n * window_overlap_accelerometer
        end = start + non_wear_window_accelerometer

        # If temp is below t0 and std below threshold, result is non-wear
        if (windowed_results[0, n] < t0) & (windowed_results[1, n] == 1):
            non_wear_vector[start:end] = 0

        # If temp is above t0, result is wear
        elif windowed_results[0, n] >= t0:
            non_wear_vector[start:end] = 1

        # If there hasn't been at least 1 window of data collected, these won't work as it requires checking against
        # the previous value
        elif n <= num_overlaps_in_window:
            non_wear_vector[start:end] = 1

        # If temperature has risen since the previous window length, result is wear
        elif windowed_results[0, n] > windowed_results[0, n - int(num_overlaps_in_window)]:
            non_wear_vector[start:end] = 1

        # If temperature has decreased since the previous window length, result is non-wear
        elif windowed_results[0, n] < windowed_results[0, n - int(num_overlaps_in_window)]:
            non_wear_vector[start:end] = 0

        # If temperature has decreased since the previous window length, result is the previous value
        elif windowed_results[0, n] == windowed_results[0, n - int(num_overlaps_in_window)]:
            non_wear_vector[start:end] = non_wear_vector[end - int(window_overlap_accelerometer)]

        else:
            raise Exception("How did you get here?")

    # Remove bouts shorter then minimum_nw_duration
    non_wear_vector = np.invert(np.array(non_wear_vector, bool))

    if not quiet:
        print("Finished Zhou Calculation.")

    return non_wear_vector

detach_nonwear = vertdetach
vert_nonwear = vertdetach


def detect_nonwear(alg='vert', **kwargs):
    """

    Detects accelerometer non-wear periods using a choice of different algorithms.


    Args:
        alg:       non-wear detection algorithm to use ('vert', 'vanhees', 'zhou', default is 'vert')
        **kwargs:

    Returns:

    """
    if alg == 'vanhees':
        print('Type not yet implemented')
        return
    elif alg == 'zhou':
        print('Type not yet implemented')
        return
    elif alg == 'vert':
        nonwear_times, nonwear_array = vert_nonwear(**kwargs)
    else:
        print('Invalid type')
        return

    return nonwear_times, nonwear_array
