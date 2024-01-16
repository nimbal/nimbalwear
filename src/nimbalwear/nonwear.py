# Adam Vert
# December 5, 2021

# ======================================== IMPORTS ========================================
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from nimbaldetach import nimbaldetach


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

detach_nonwear = nimbaldetach
vert_nonwear = nimbaldetach

def detect_nonwear(alg='detach', **kwargs):
    """

    Detects accelerometer non-wear periods using a choice of different algorithms.


    Args:
        alg:       non-wear detection algorithm to use ('detach', 'vanhees', 'zhou', default is 'vert')
        **kwargs:

    Returns:

    """
    if alg == 'vanhees':
        print('Type not yet implemented')
        return
    elif alg == 'zhou':
        print('Type not yet implemented')
        return
    elif alg == 'detach':
        nonwear_times, nonwear_array = detach_nonwear(**kwargs)
    else:
        print('Invalid type')
        return

    return nonwear_times, nonwear_array

def nonwear_stats(nonwear_bouts, sum_type='daily', quiet=False):
    """
    Calculates summary for each date in collection.

    @param nonwear_bouts:
    @param sum_type:
    @param coll_start: datetime of collection start
    @param coll_end: datetime of collection end
    @param quiet:

    @return:
    """



    """
    """

    nonwear_bouts = deepcopy(nonwear_bouts)

    # create date and duration columns
    nonwear_bouts['date'] = pd.to_datetime(nonwear_bouts['start_time']).dt.date
    nonwear_bouts['duration'] = (nonwear_bouts['end_time'] - nonwear_bouts['start_time']).dt.total_seconds().round()

    nonwear_stats = None

    # check summary type
    if sum_type == 'daily':

        # DAILY SUMMARY
        if not quiet:
            print("Summarizing daily wear and non-wear time...")

        #collect_stats = pd.DataFrame(columns=['day_num', 'date', 'collect'])
        nonwear_stats = pd.DataFrame(columns=['day_num', 'date', 'wear', 'nonwear'])

        # COLLECTION TIME - calculate known collection time for each day

        # calculate range of dates with known collection time
        # start_date = (min(nonwear_bouts['start_time']) + timedelta(days=1) if coll_start is None else coll_start).date()
        # end_date = (max(nonwear_bouts['end_time']) - timedelta(days=1) if coll_end is None else coll_end).date()
        # dates = pd.date_range(start_date, end_date)
        #
        # # calculate collection time for each day
        # collect = []
        # for date in dates:
        #     day_start = datetime.combine(date, datetime.min.time())
        #     day_end = datetime.combine(date + timedelta(days=1), datetime.min.time())
        #     if coll_start is not None:
        #         day_start = coll_start if (date.date() == coll_start.date()) else day_start
        #     if coll_end is not None:
        #         day_end = coll_end if date.date() == coll_end.date() else day_end
        #     duration = round((day_end - day_start).total_seconds())
        #     collect.append(duration)
        #
        # # create collection time dataframe
        # collect_stats = pd.DataFrame({'date': [d.date() for d in dates], 'collect': collect})

        # NON-WEAR

        # if bout crosses midnight then split into two bouts
        new_bouts = []
        for i, r in nonwear_bouts.iterrows():

            # if row ends on next day
            if r['date'] != pd.to_datetime(r['end_time']).date():

                # find midnight
                first_midnight = last_midnight = datetime.combine(r['start_time'].date() + timedelta(days=1),
                                                                  datetime.min.time())

                # add full nonwear days if bout was longer than one full day
                full_days = pd.date_range(r['start_time'].date() + timedelta(days=1),
                                          r['end_time'].date() - timedelta(days=1))
                for date in full_days.date:
                    # calculate start and end datetime and append new row
                    start_midnight = datetime.combine(date, datetime.min.time())
                    end_midnight = datetime.combine(date + timedelta(days=1), datetime.min.time())
                    new_bouts.append([r['id'], r['event'], start_midnight, end_midnight, date,
                                      round((end_midnight - start_midnight).total_seconds())])
                    last_midnight = end_midnight

                # create new bout from last midnight to end
                new_bouts.append([r['id'], r['event'], last_midnight, r['end_time'], r['end_time'].date(),
                                  round((r['end_time'] - last_midnight).total_seconds())])

                # adjust current bout to end at midnight
                nonwear_bouts.at[i, 'end_time'] = first_midnight
                nonwear_bouts.at[i, 'duration'] = round((first_midnight - r['start_time']).total_seconds())

        # add new bouts
        for new_bout in new_bouts:
            new_row = pd.DataFrame([new_bout], columns=nonwear_bouts.columns)
            nonwear_bouts = pd.concat([nonwear_bouts, new_row], ignore_index=True)

        # loop through days and calculate nonwear time
        # dates = []
        # nonwear = []

        day_num = 1
        for date, date_group in nonwear_bouts.groupby('date'):

            counts = {'wear': 0, 'nonwear': 0, }

            for event, event_group in date_group.groupby('event'):
                counts[event] = round(sum(event_group['duration']))

            day_nonwear_stats = pd.DataFrame([[day_num, date, counts['wear'], counts['nonwear']]],
                                             columns=nonwear_stats.columns)

            nonwear_stats = pd.concat([nonwear_stats, day_nonwear_stats], ignore_index=True)

            day_num += 1

        #     dates.append(date)
        #     duration = sum(date_group['duration'])
        #     nonwear.append(duration)
        #
        # # create nonwear dataframe
        # nonwear_stats = pd.DataFrame({'date': dates, 'nonwear': nonwear})
        #
        # # merge collection time and nonwear time dataframes by date
        # nonwear_stats = pd.merge(collect_stats, nonwear_stats, how='outer', on='date', sort=True)
        #
        # # calculate wear time
        # nonwear_stats['nonwear'] = nonwear_stats['nonwear'].fillna(0)
        # nonwear_stats['wear'] = nonwear_stats['collect'] - nonwear_stats['nonwear']
        #
        # nonwear_stats = nonwear_stats.astype(
        #     {'collect': pd.Int64Dtype(), 'wear': pd.Int64Dtype(), 'nonwear': pd.Int64Dtype(), })
        #
        # nonwear_stats['day_num'] = range(1, nonwear_stats.shape[0] + 1)
        # nonwear_stats = nonwear_stats[['day_num', 'date', 'collect', 'wear', 'nonwear']]

    else:
        print('Invalid sum_type selected.')

    return nonwear_stats