"""
The following algorithm is to detect the sleep period time window (SPT-window) which is the time window starting at
sleep onset and ending when waking up after the last sleep episode of the night.
It detects sleep without the use of a sleep diary.

Step 1: Use accelerometer data to find sustained inactivity bouts
    - non-wear was removed before calculating the 10th percentile
    - using the z-angle, keeps periods that are below the threshold which last for minimum 30 minutes
    - if gaps between periods are less than 60 minutes, periods are combined
Step 2: Use temperature data to find periods of elevated temperature
    - non-wear was removed before calculating the mean temperature
Step 3: Find periods of overlap from accelerometer and temperature data
Step 4: Remove periods of non-wear

The following code is based on the following Algorithm described in a paper: van Hees 2018 - Heuristic algorithm looking at Distribution of Change in Z-Angle (HDCZA)

"""
import math
import datetime as dt

import numpy as np
import pandas as pd


def z_angle_change(x_values, y_values, z_values, epoch_samples):
    """

    Args:
        x_values:           accelerometer x values (g)
        y_values:           accelerometer y values (g)
        z_values:           accelerometer z values (g)
        epoch_samples:      number of samples per epoch

    Returns:
        z_angle:            Consecutive 5-second average z-angle (HDCZA Step 3)
        z_angle_change:     Absolute change in consecutive 5-second average z-angle (HDCZA Step 4)

    """
    ####################################################
    # HDCZA Step 1: Calculate 5-second rolling medians
    ####################################################

    x_rolling_median = pd.Series(x_values).rolling(epoch_samples).median()
    y_rolling_median = pd.Series(y_values).rolling(epoch_samples).median()
    z_rolling_median = pd.Series(z_values).rolling(epoch_samples).median()

    ##############################################################
    # HDCZA Step 2: Calculate z-angle per 5-second rolling window
    ##############################################################

    # calculate z-angle
    z_angle = (np.arctan(z_rolling_median / np.sqrt(np.square(x_rolling_median) + np.square(y_rolling_median)))) * 180 / np.pi

    # adjust so index is median for next raw_epoch_length instead of previous raw_epoch_length
    z_angle = z_angle[epoch_samples - 1:].tolist()

    #########################################################
    # HDCZA Step 3: Calculate consecutive 5-second averages
    #########################################################

    z_angle = [np.mean(z_angle[i:i + epoch_samples]) for i in range(0, len(z_angle), epoch_samples)]

    ###########################################################################
    # HDCZA Step 4: Absolute difference between consecutive 5-second averages
    ###########################################################################

    z_angle_change = [0] + abs(np.diff(z_angle))

    return z_angle, z_angle_change

def detect_sptw(x_values, y_values, z_values, sample_rate, start_datetime, nonwear=None, day_offset=12, raw_epoch_length=5,
                 z_epoch_length=300, min_wear_hours=3, min_sptw_length=30, max_gap_time=60, z_per_threshold=10):
    '''

    Uses Heuristic algorithm looking at Distribution of Change in Z-Angle (HDCZA) to detect sleep period time
    windows (sptw). Detects all candidates, not just longest.

    van Hees et al. (2018). Estimating sleep parameters using an accelerometer without sleep diary.
    Scientific Reports.

    Use sustained inactivity to detect sleep episodes in each sptw

    van Hees et al. (2015). A novel, open-access method to assess sleep duration using a wrist-worn
    accelerometer. PLoS ONE.

    Args:
        x_values:           accelerometer x values (g)
        y_values:           accelerometer y values (g)
        z_values:           accelerometer z values (g) - must be perpendicular to skin surface
        sample_rate:        samples per second (Hz)
        start_datetime:     timestamp of first data point (datetime object)
        day_offset:         offset from midnight for start of days when grouping by day (hours)
        raw_epoch_length:   length of epoch for rolling median and z angle averages (HDCZA steps 1-3) (seconds, default = 5)
        z_epoch_length:     length of epoch for rolling median of z angle change (HDCZA step 5) (seconds, default = 300)
        min_sptw_length:    minimum length of sptw (minutes, default = 30)
        max_gap_time:       maximum gap between sptw to be considered continuous (minutes, default = 60)
        z_per_threshold:    percentile threshold of z angle change used in sptw detection (percentile; default = 10)



    Returns:
        sptw:               pandas DataFrame containing index and time data for each sptw
        sleep_bouts:        pandas DataFrame containing index and time data for each sleep bout

    '''

    # translate epoch seconds to samples
    raw_epoch_samples = sample_rate * raw_epoch_length
    z_epoch_samples = sample_rate * z_epoch_length

    ###############################################
    # HDCZA Steps 1-4: Calculate change in z angle
    ###############################################

    z_angle, z_angle_diff = z_angle_change(x_values, y_values, z_values, raw_epoch_samples)

    #######################################################
    # HDCZA Step 5: Rolling median using 5-minute window
    ######################################################

    # calculate rolling median
    z_angle_diff_med = pd.Series(z_angle_diff).rolling(round(z_epoch_samples / raw_epoch_samples)).median()

    # adjust so index is median for next z_epoch_length instead of previous z_epoch_length
    z_angle_diff_med = z_angle_diff_med[round(z_epoch_samples / raw_epoch_samples) - 1:].tolist()

    z_num_samples = len(z_angle_diff_med)
    z_sample_rate = 1 / raw_epoch_length

    ########################################
    # negate nonwear if it was passed in
    ########################################

    if nonwear is not None:

        nonwear = nonwear.copy()

        # calculate nonwear start and end datapoints
        nonwear_start_dp = [round((x - start_datetime).total_seconds() * z_sample_rate) for x in nonwear['start_time']]
        nonwear_end_dp = [round((x - start_datetime).total_seconds() * z_sample_rate) for x in nonwear['end_time']]

        # set all datapoints in nonwear bout to -9
        for i in range(len(nonwear_start_dp)):
            z_angle_diff_med[nonwear_start_dp[i]:nonwear_end_dp[i]] = [-9] * (nonwear_end_dp[i] - nonwear_start_dp[i])


    ################################################################
    # Loop through days to perform remainning HDCZA steps each day
    ################################################################

    # calculate day start sample indices
    first_day_start = dt.datetime.combine(start_datetime - dt.timedelta(hours=day_offset), dt.time.min) \
                      + dt.timedelta(hours=day_offset)
    days = ((z_num_samples / z_sample_rate) + (start_datetime - first_day_start).total_seconds()) / (60 * 60 * 24)

    day_start_times = [first_day_start + dt.timedelta(days=x) for x in range(math.ceil(days))]
    day_start_times[0] = start_datetime

    day_start_indices = [round((day_start_time - start_datetime).total_seconds() * z_sample_rate) for day_start_time
                         in day_start_times]

    sptw_start_time = []
    sptw_end_time = []

    # loop through each day
    for i in range(len(day_start_indices)):

        start_index = day_start_indices[i]
        end_index = day_start_indices[i + 1] if (i + 1) < len(day_start_indices) else None

        # Calculate wearable data in current day
        z_day = np.array(z_angle_diff_med[start_index:end_index])
        z_day_wear = [x for x in z_day if x >=0]

        # Proceed if minimum wearable data criteria met
        if len(z_day_wear) > (min_wear_hours * 3600 * z_sample_rate):

            #########################################################
            # HDCZA Step 6: Detect values < (10th percentile * 15)
            #########################################################

            # caclulate z-angle threshold value
            z_angle_threshold = np.percentile(a=z_day_wear, q=z_per_threshold) * 15

            # generate boolean indicating if each data point is below threshold
            z_below_per_threshold = ((z_day >= 0) & (z_day < z_angle_threshold)).tolist()

            ###########################################
            # HDCZA Step 7: Keep blocks > 30 minutes
            ###########################################

            # detect where data points cross threshold
            z_angle_cross_threshold = [j for j in range(1, len(z_below_per_threshold))
                                       if z_below_per_threshold[j] != z_below_per_threshold[j - 1]]

            # only continue if sptw candidates detected (i.e., some values above and below threshold)
            if z_angle_cross_threshold:

                # if start or end of collection could be during sptw then add start or end index
                if z_below_per_threshold[z_angle_cross_threshold[0]] == False:
                    z_angle_cross_threshold.insert(0, 0)
                if z_below_per_threshold[z_angle_cross_threshold[-1]] == True:
                    z_angle_cross_threshold.append(len(z_below_per_threshold) - 1)

                # generate sptw candidate list from threshold crossings (all windows below threshold)
                sptw_candidates = np.reshape(z_angle_cross_threshold, (math.floor(len(z_angle_cross_threshold) / 2), 2)).tolist()

                sptw_del = []
                j = 0

                # detect sptw candidates shorter than minimum length (30 minutes default)
                for s in sptw_candidates:
                    if (s[1] - s[0]) < (min_sptw_length * 60 / raw_epoch_length):
                        sptw_del.append(j)
                    j += 1

                # delete these short windows
                sptw_candidates = np.delete(sptw_candidates, sptw_del, 0).T.tolist()

                #################################################
                # HDCZA Step 8: Include time gaps < 60 minutes
                #################################################

                # detect windows that have a gap less than max_gap_time (default 60 minutes) if no nonwear between
                sptw_del = []
                for j in range(1, len(sptw_candidates[0])):
                    if (((sptw_candidates[0][j] - sptw_candidates[1][j - 1]) < (max_gap_time * 60 / raw_epoch_length))
                            & (all(y >= 0 for y in z_day[sptw_candidates[1][j - 1]:sptw_candidates[0][j]]))):
                        sptw_del.append(j)

                # delete end point of first window and start point of second window to combine into one window
                for j in sorted(sptw_del, reverse=True):
                    del sptw_candidates[0][j]
                    del sptw_candidates[1][j - 1]

                sptw_start_time.extend([start_datetime + dt.timedelta(seconds=((start_index + j) * raw_epoch_length))
                                        for j in sptw_candidates[0]])
                sptw_end_time.extend([start_datetime + dt.timedelta(seconds=((start_index + j) * raw_epoch_length))
                                      for j in sptw_candidates[1]])

    ##########################
    # Format data for return
    ##########################

    sptw_date = [(x - dt.timedelta(hours=day_offset)).date()for x in sptw_start_time]

    sptw = {'sptw_num': list(range(1, len(sptw_start_time) + 1)),
            'relative_date': sptw_date,
            'start_time': sptw_start_time,
            'end_time': sptw_end_time}

    sptw = pd.DataFrame.from_dict(sptw)

    return sptw, z_angle, z_angle_diff, z_sample_rate


def detect_sleep_bouts(z_angle_diff, sptw, z_sample_rate, start_datetime, raw_epoch_length=5, z_abs_threshold=5, min_sleep_length=5):
    '''

    Args:
        z_angle_diff:
        z_sample_rate:
        start_datetime:
        raw_epoch_length:
        z_abs_threshold:    absolute threshold of z angle change used in sleep bout detection (degrees; default = 5)
        min_sleep_length:   minimum length of sleep bout (minutes; default = 5)

    Returns:

    '''

    sptw = sptw.copy()

    z_sptw_start_indices = [round((sptw_start_time - start_datetime).total_seconds() * z_sample_rate)
                            for sptw_start_time in sptw['start_time']]

    z_sptw_end_indices = [round((sptw_end_time - start_datetime).total_seconds() * z_sample_rate)
                            for sptw_end_time in sptw['end_time']]

    sleep_sptw_num = []
    sleep_start_time = []
    sleep_end_time = []

    # loop through each sptw
    for i in range(len(sptw['sptw_num'])):

        start_index = z_sptw_start_indices[i]
        end_index = z_sptw_end_indices[i]

        # Detect when z-angle values below threshold
        z_below_per_threshold = (z_angle_diff[start_index:end_index] <= z_abs_threshold).tolist()

        # Keep blocks > min_sleep_length

        z_angle_cross_threshold = [j for j in range(1, len(z_below_per_threshold))
                                   if z_below_per_threshold[j] != z_below_per_threshold[j - 1]]

        # only continue if sleep bout candidates detected (i.e., some values above and below threshold)
        if z_angle_cross_threshold:

            if z_below_per_threshold[z_angle_cross_threshold[0]] == False:
                z_angle_cross_threshold.insert(0, 0)
            if z_below_per_threshold[z_angle_cross_threshold[-1]] == True:
                z_angle_cross_threshold.append(len(z_below_per_threshold) - 1)

            sleep_candidates = np.reshape(z_angle_cross_threshold,
                                          (math.floor(len(z_angle_cross_threshold) / 2), 2)).tolist()

            sleep_del = []
            j = 0

            for s in sleep_candidates:
                if (s[1] - s[0]) < (min_sleep_length * 60 / raw_epoch_length):
                    sleep_del.append(j)
                j += 1

            sleep_candidates = np.delete(sleep_candidates, sleep_del, 0).T.tolist()

            sleep_sptw_num.extend([sptw['sptw_num'][i]] * len(sleep_candidates[0]))
            sleep_start_time.extend([start_datetime + dt.timedelta(seconds=((start_index + j) * raw_epoch_length))
                                     for j in sleep_candidates[0]])
            sleep_end_time.extend([start_datetime + dt.timedelta(seconds=((start_index + j) * raw_epoch_length))
                                   for j in sleep_candidates[1]])

    sleep_bouts = {'sleep_bout_num': list(range(1, len(sleep_start_time) + 1)),
                   'sptw_num': sleep_sptw_num,
                   'start_time': sleep_start_time,
                   'end_time': sleep_end_time}

    sleep_bouts = pd.DataFrame.from_dict(sleep_bouts)

    return sleep_bouts


def detect_sleep(x_values, y_values, z_values, sample_rate, start_datetime, nonwear=None, day_offset=12,
                 raw_epoch_length=5, z_epoch_length=300, min_wear_hours=3, min_sptw_length=30, max_gap_time=60,
                 z_per_threshold=10, z_abs_threshold=5, min_sleep_length=5):

    sptw, z_angle, z_angle_diff, z_sample_rate = detect_sptw(x_values=x_values, y_values=y_values, z_values=z_values,
                                                             sample_rate=sample_rate, start_datetime=start_datetime,
                                                             nonwear=nonwear, day_offset=day_offset,
                                                             raw_epoch_length=raw_epoch_length,
                                                             z_epoch_length=z_epoch_length,
                                                             min_wear_hours=min_wear_hours,
                                                             min_sptw_length=min_sptw_length,
                                                             max_gap_time=max_gap_time, z_per_threshold=z_per_threshold)

    sleep_bouts = detect_sleep_bouts(z_angle_diff=z_angle_diff, sptw=sptw, z_sample_rate=z_sample_rate,
                                     start_datetime=start_datetime, raw_epoch_length=raw_epoch_length,
                                     z_abs_threshold=z_abs_threshold, min_sleep_length=min_sleep_length)

    return sptw, sleep_bouts, z_angle, z_angle_diff, z_sample_rate


def sptw_stats(sptw, sleep_bouts, type='daily', sptw_inc='long'):
    """

    Args:
        sptw:
        sleep_bouts:
        type:           'all', 'daily'
        sptw_inc:       types of sptw to include in daily summary ('long' = longest sptw, 'all' = all sptws, 'sleep' =
                        all sptws that contain sleep bouts) (can specify multiple in a list to run multiple summaries)

    Returns:

    """

    sptw = sptw.copy()
    sptw['start_time'] = pd.to_datetime(sptw['start_time'], format='%Y-%m-%d %H:%M:%S')
    sptw['end_time'] = pd.to_datetime(sptw['end_time'], format='%Y-%m-%d %H:%M:%S')
    sptw['duration'] = [round((y - x).total_seconds() / 60) for (x, y) in zip(sptw['start_time'], sptw['end_time'])]

    sleep_bouts = sleep_bouts.copy()
    sleep_bouts['start_time'] = pd.to_datetime(sleep_bouts['start_time'], format='%Y-%m-%d %H:%M:%S')
    sleep_bouts['end_time'] = pd.to_datetime(sleep_bouts['end_time'], format='%Y-%m-%d %H:%M:%S')
    sleep_bouts['duration'] = [round((y - x).total_seconds() / 60)
                               for (x, y) in zip(sleep_bouts['start_time'], sleep_bouts['end_time'])]

    sptw_inc = [sptw_inc] if not isinstance(sptw_inc, list) else sptw_inc

    sleep_stats = None

    if type == 'daily':

        sleep_stats = pd.DataFrame(columns=['day_num', 'date', 'type', 'sptw_inc', 'sptw_duration', 'sleep_duration',
                                            'sleep_to_wake_duration', 'se', 'waso'])

        # get unique dates in sptw
        dates = sptw['relative_date'].unique()

        for s in sptw_inc:

            day_num = 1

            # loop through each unique date
            for date in dates:

                total_sleep_duration = 0
                total_sptw_duration = 0
                sleep_to_wake_duration = 0

                # get all sptw for current date
                sptw_day = sptw.loc[sptw['relative_date'] == date]

                # get longest sptw if 'daily_long' type is selected
                sptw_day = sptw_day.iloc[[sptw_day['duration'].argmax()]] if s == 'long' else sptw_day

                for i, sptw_cur in sptw_day.iterrows():

                    # get all sleep_bouts in current sptw
                    sleep_bouts_sptw = sleep_bouts.loc[sleep_bouts['sptw_num'] == sptw_cur['sptw_num']]
                    sleep_bouts_sptw.reset_index(inplace=True)

                    # if current sptw has sleep bouts in it
                    if not sleep_bouts_sptw.empty:

                        # get sptw duration
                        total_sptw_duration += sptw_cur['duration']
                        total_sleep_duration += sum(sleep_bouts_sptw['duration'])

                        # calculate sleep to wake duration (first sleep bout start to last sleep bout end)
                        sleep_onset_time = sleep_bouts_sptw.iloc[0]['start_time']
                        waking_time = sleep_bouts_sptw.iloc[-1]['end_time']
                        sleep_to_wake_duration += round((waking_time - sleep_onset_time).total_seconds() / 60)

                    # if current sptw has no sleep bouts
                    else:

                        # if sptw_type is long or all then add sptw duration
                        if s in ['long', 'all']:

                            # get sptw duration
                            total_sptw_duration += sptw_cur['duration']

                # calculate sleep efficiency (total duration sleeping / total sptw length
                sleep_eff = round(total_sleep_duration / total_sptw_duration, 4) if total_sptw_duration != 0 else 0

                # calculate wakefulness after sleep onset (time not sleeping between first sleep start and last sleep end)
                waso = sleep_to_wake_duration - total_sleep_duration

                day_sleep_stats = pd.DataFrame({'day_num': day_num,
                                                'date': date,
                                                'type': type,
                                                'sptw_inc': s,
                                                'sptw_duration': total_sptw_duration,
                                                'sleep_duration': total_sleep_duration,
                                                'sleep_to_wake_duration': sleep_to_wake_duration,
                                                'se': sleep_eff,
                                                'waso': waso})

                sleep_stats = pd.concat([sleep_stats, day_sleep_stats], ignore_index=True)

                day_num += 1

    elif type == 'all':

        sleep_stats = pd.DataFrame(columns=['sptw_num', 'start_time', 'end_time', 'type', 'sptw_duration',
                                            'sleep_duration', 'sleep_to_wake_duration', 'se', 'waso'])

        # loop through each sptw
        for i, sptw_cur in sptw.iterrows():

            total_sleep_duration = 0
            sptw_duration = 0
            sleep_to_wake_duration = 0

            # get all sleep_bouts in current sptw
            sleep_bouts_sptw = sleep_bouts.loc[sleep_bouts['sptw_num'] == sptw_cur['sptw_num']]
            sleep_bouts_sptw.reset_index(inplace=True)

            # get sptw duration
            sptw_duration = sptw_cur['duration']

            # get total sleep duration
            total_sleep_duration = sum(sleep_bouts_sptw['duration'])

            # if sleep bouts found, calculate sleep to wake duration (first sleep bout start to last sleep bout end)
            if not sleep_bouts_sptw.empty:
                sleep_onset_time = sleep_bouts_sptw.iloc[0]['start_time']
                waking_time = sleep_bouts_sptw.iloc[-1]['end_time']
                sleep_to_wake_duration = round((waking_time - sleep_onset_time).total_seconds() / 60)
            else:
                sleep_to_wake_duration = 0

            # calculate sleep efficiency (total duration sleeping / total sptw length
            sleep_eff = round(total_sleep_duration / sptw_duration, 4) if sptw_duration != 0 else 0

            # calculate wakefulness after sleep onset (time not sleeping between first sleep start and last sleep end)
            waso = sleep_to_wake_duration - total_sleep_duration

            sleep_stats = sleep_stats.append({'sptw_num': sptw_cur['sptw_num'],
                                              'start_time': sptw_cur['start_time'],
                                              'end_time': sptw_cur['end_time'],
                                              'type': type,
                                              'sptw_duration': sptw_duration,
                                              'total_sleep_duration': total_sleep_duration,
                                              'sleep_to_wake_duration': sleep_to_wake_duration,
                                              'se': sleep_eff,
                                              'waso': waso},
                                             ignore_index=True)
    else:
        print('Invalid type selected.')

    return sleep_stats
