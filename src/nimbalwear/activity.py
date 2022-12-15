from collections import Counter
from datetime import timedelta, datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt


def avm_cutpoints(cutpoint_type='Powell', dominant=False):

    """
    Powell


    Fraysse, F., Post, D., Eston, R., Kasai, D., Rowlands, A. v., & Parfitt, G. (2021). Physical Activity Intensity
    Cut-Points for Wrist-Worn GENEActiv in Older Adults. Frontiers in Sports and Active Living, 2.
    https://doi.org/10.3389/fspor.2020.579278


    """

    cutpoints_dict = {'Powell': {'dominant': {'light': 0.1133, 'moderate': 0.1511, 'vigorous': 0.3156},
                                 'non-dominant': {'light': 0.1044, 'moderate': 0.1422, 'vigorous': 0.3489}},
                      'Fraysse': {'dominant': {'light': 0.0625, 'moderate': 0.0925},
                                  'non-dominant': {'light': 0.0425, 'moderate': 0.098}}}

    dom = 'dominant' if dominant else 'non-dominant'

    cutpoints = cutpoints_dict[cutpoint_type][dom]

    return cutpoints


def activity_wrist_avm(x, y, z, sample_rate, start_datetime, lowpass=20, epoch_length=15, cutpoint='Powell',
                       dominant=False, sedentary_gait=False, gait=pd.DataFrame(), nonwear=pd.DataFrame(),
                       sptw=pd.DataFrame(), sleep_bouts=pd.DataFrame(), quiet=False):

    """
    Transforms Powell cutpoints to avm based on their 15 second epochs at 30 Hz, probably most
    accurate at similar values.

    returns avm (in mg) and intensity for each epoch

    """

    accel = np.vstack((x, y, z))
    epoch_samples = int(epoch_length * sample_rate)
    sec_samples = int(sample_rate)

    if lowpass is not None:

        if not quiet:
            print("Applying lowpass filter to data...")

        # low-pass filter
        order = 2
        sos = butter(N=order, Wn=lowpass, btype='lowpass', analog=False, output='sos', fs=sample_rate)
        accel = sosfilt(sos, accel)

    # calculate vector magnitudes
    if not quiet:
        print("Calculating vector magnitudes...")

    vm = np.sqrt(np.square(accel).sum(axis=0)) - 1
    vm[vm < 0] = 0

    # calculate avm
    if not quiet:
        print("Calculating average vector magnitude for each epoch...")

    #epoch_starts = range(0, len(vm) + 1 - epoch_samples, epoch_samples)
    avm = vm[:int(len(vm) / epoch_samples) * epoch_samples].reshape(-1, epoch_samples).sum(axis=1) / epoch_samples
    avm_sec = vm[:int(len(vm) / sec_samples) * sec_samples].reshape(-1, sec_samples).sum(axis=1) / sec_samples
    avm_sec = np.around(avm_sec * 1000, 2)

    cutpoints = avm_cutpoints(cutpoint, dominant)

    # classify intensity by comparing avm to cutpoints
    epoch_intensity = ['sedentary'] * len(avm)
    for k, v in cutpoints.items():
        epoch_intensity = [k if x >= v else epoch_intensity[i] for i, x in enumerate(avm)]

    # initialize activity_epochs DataFrame
    activity_epochs = pd.DataFrame({'activity_epoch_num': range(1, len(avm) + 1)})

    # add start and end time for each epoch
    if start_datetime is not None:
        activity_epochs['start_time'] = [start_datetime + timedelta(seconds=(int(x) - 1) * epoch_length)
                                         for x in activity_epochs['activity_epoch_num']]
        activity_epochs['end_time'] = [start_datetime + timedelta(seconds=(int(x)) * epoch_length)
                                       for x in activity_epochs['activity_epoch_num']]

    # add avm and intensity for each epoch
    activity_epochs['avm'] = np.around(avm * 1000, 2)
    activity_epochs['intensity'] = epoch_intensity

    if not quiet:
        print("Removing non-wear and sleep if entered...")

    # set intensity to 'none' during nonwear
    for idx, row in nonwear.iterrows():

        activity_epochs.loc[((activity_epochs['start_time'] < row['end_time'])
                             & (activity_epochs['end_time'] > row['start_time'])), 'intensity'] = 'none'

    # set intensity to 'none' during all sptw that contain sleep
    if not (sptw.empty or sleep_bouts.empty):
        sptw = sptw.loc[sptw['sptw_num'].isin(sleep_bouts['sptw_num'].unique())]

    for idx, row in sptw.iterrows():
        activity_epochs.loc[((activity_epochs['start_time'] < row['end_time'])
                             & (activity_epochs['end_time'] > row['start_time'])), 'intensity'] = 'none'

    # set 'sedentary' intensity to 'sedentary_gait' during gait
    if sedentary_gait:
        for idx, row in gait.iterrows():
            activity_epochs.loc[((activity_epochs['intensity'] == 'sedentary')
                                 & (activity_epochs['start_time'] < row['end_time'])
                                 & (activity_epochs['end_time'] > row['start_time'])), 'intensity'] = 'sedentary_gait'


    # create bouts
    epoch_intensity = activity_epochs['intensity']

    bout_starts = [i for i in range(1, len(epoch_intensity))
                   if epoch_intensity[i] != epoch_intensity[i - 1]]
    bout_starts.insert(0, 0)

    bout_ends = bout_starts[1:]
    bout_ends.append(len(epoch_intensity))

    bout_intensity = [epoch_intensity[i] for i in bout_starts]

    bout_start_times = [start_datetime + timedelta(seconds=(int(x)) * epoch_length) for x in bout_starts]
    bout_end_times = [start_datetime + timedelta(seconds=(int(x)) * epoch_length) for x in bout_ends]

    activity_bouts = pd.DataFrame({'activity_bout_num': np.arange(1, len(bout_starts) + 1),
                                   'start_time': bout_start_times,
                                   'end_time': bout_end_times,
                                   'intensity': bout_intensity})

    return activity_epochs, activity_bouts, avm, vm, avm_sec


def sum_total_activity(epoch_intensity, epoch_length, quiet=False):

    if not quiet:
        print("Summarizing total activity...")

    epoch_per_min = 60 / epoch_length

    epoch_intensity_counts = Counter(epoch_intensity)

    none = epoch_intensity_counts['none'] / epoch_per_min if 'none' in epoch_intensity_counts.keys() else 0
    sedentary = epoch_intensity_counts['sedentary'] / epoch_per_min \
        if 'sedentary' in epoch_intensity_counts.keys() else 0
    light = epoch_intensity_counts['light'] / epoch_per_min if 'light' in epoch_intensity_counts.keys() else 0
    moderate = epoch_intensity_counts['moderate'] / epoch_per_min if 'moderate' in epoch_intensity_counts.keys() else 0
    vigorous = epoch_intensity_counts['vigorous'] / epoch_per_min if 'vigorous' in epoch_intensity_counts.keys() else 0

    return pd.DataFrame({'none': none, 'sedentary': sedentary, 'light': light, 'moderate': moderate, 'vigorous': vigorous})


def activity_stats(activity_epochs, type='daily', quiet=False):

    #TODO: make generic event stats (activity_stats and nonwear stats identical)

    """Calculates summary for each date in collection.

    Includes all epochs that start in that day for that day.
    """
    activity_epochs = deepcopy(activity_epochs)

    activity_epochs['date'] = pd.to_datetime(activity_epochs['start_time']).dt.date
    activity_epochs['duration'] = [round((x['end_time'] - x['start_time']).total_seconds())
                                   for i, x in activity_epochs.iterrows()]

    activity_epochs.drop('avm', axis=1, inplace=True, errors='ignore')

    activity_stats = None

    if type == 'daily':

        if not quiet:
            print("Summarizing daily activity...")

        activity_stats = pd.DataFrame(columns=['day_num', 'date', 'none', 'sedentary', 'sedentary_gait', 'light',
                                               'moderate', 'vigorous'])


        new_epochs = []
        for idx, row in activity_epochs.iterrows():
            if row['date'] != pd.to_datetime(row['end_time']).date():
                midnight = datetime.combine(pd.to_datetime(row['end_time']).date(), datetime.min.time())
                new_epochs.append([0, midnight, row['end_time'], row['intensity'], row['end_time'].date(),
                                   round((row['end_time'] - midnight).total_seconds())])
                activity_epochs.at[idx, 'end_time'] = midnight
                activity_epochs.at[idx, 'duration'] = round((midnight - row['start_time']).total_seconds())


        for new_epoch in new_epochs:
            new_row = pd.DataFrame([new_epoch], columns=activity_epochs.columns)
            activity_epochs = pd.concat([activity_epochs, new_row], ignore_index=True)

        #activity_epochs.sort_values(by='start_time', inplace=True, ignore_index=True)

        day_num = 1

        for date, date_group in activity_epochs.groupby('date'):

            counts = {'none': 0, 'sedentary': 0, 'sedentary_gait': 0, 'light': 0, 'moderate': 0, 'vigorous': 0}

            for intensity, intensity_group in date_group.groupby('intensity'):
                counts[intensity] = round(sum(intensity_group['duration']) / 60, 2)

            day_activity_stats = pd.DataFrame([[day_num, date, counts['none'], counts['sedentary'],
                                                counts['sedentary_gait'],counts['light'], counts['moderate'],
                                                counts['vigorous']]], columns=activity_stats.columns)

            activity_stats = pd.concat([activity_stats, day_activity_stats], ignore_index=True)

            day_num += 1

    else:
        print('Invalid type selected.')

    return activity_stats


# def calculate_wrist_activity(file_path, epoch_length, dominant=False, quiet=False):
#
#     # NOT FULLY FUNCTIONAL
#
#     # read file
#
#     wrist_device = nwdata.NWData()
#     wrist_device.import_edf(file_path)
#
#     # search for appropriate signals
#
#     avm, epoch_intensity = calc_wrist_powell(x=wrist_device.signals[0],
#                                              y=wrist_device.signals[1],
#                                              z=wrist_device.signals[2],
#                                              sample_rate=wrist_device.signal_headers[0]['sample_rate'],
#                                              epoch_length=epoch_length,
#                                              dominant=dominant,
#                                              quiet=quiet)
#
#     total_activity = sum_total_activity(epoch_intensity=epoch_intensity, epoch_length=epoch_length, quiet=quiet)
#     daily_activity = sum_daily_activity(epoch_intensity, epoch_length=epoch_length,
#                                         start_datetime=wrist_device.header['startdate'], quiet=quiet)
#
#     # can add write to file option
#
#     # can add identifiers from file header if wanted before returning summaries
#
#     return avm, epoch_intensity, total_activity, daily_activity
