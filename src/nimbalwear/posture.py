import pandas as pd
import numpy as np
from scipy import signal
from datetime import timedelta as td

def prep_accelerometer_df(x, y, z, sample_rate, body_location, start_datetime):
    """
    Creates the proper dataframe format for an individual accelerometer for the remainder of the posture analysis
    """
    # Get data
    if body_location == 'chest':
        data_dict = {
            "Anterior": [-i for i in z],
            "Up": x,
            "Left": y,
            "start_stamp": start_datetime,
            "sample_rate": sample_rate}
    elif body_location in ['ankle', 'thigh']:
        data_dict = {
            "Anterior": y,
            "Up": x,
            "Left": z,
            "start_stamp": start_datetime,
            "sample_rate": sample_rate}
    elif body_location == 'wrist':
        data_dict = {
            "Anterior": [-i for i in x],
            "Up": y,
            "Left": z,
            "start_stamp": start_datetime,
            "sample_rate": sample_rate}

    # Epoch data
    ts = pd.date_range(data_dict['start_stamp'], periods=len(data_dict['Anterior']),
                       freq=(str(1 / data_dict['sample_rate']) + 'S'))
    df = pd.DataFrame({'timestamp': ts, 'ant_acc': data_dict['Anterior'],
                       'up_acc': data_dict['Up'], 'left_acc': data_dict['Left']})
    return df.resample(
        f'{1}S', on='timestamp').mean().reset_index(col_fill='timestamp')  # Resample to 1s intervals using mean


def get_gait_mask(gait_df, start, end):
    """
    creates a list of 0's (no gait) and 1's (gait) with each value representing 1 second.
    """
    # Read in gait_object

    gait_df['start_timestamp'] = pd.DatetimeIndex(gait_df['start_timestamp']).floor('S')
    gait_df['end_timestamp'] = pd.DatetimeIndex(gait_df['end_timestamp']).ceil('S')

    # Crop gait df to start and end times
    if gait_df.loc[gait_df.index[0], 'start_timestamp'] < start:
        gait_df.loc[gait_df.index[0], 'start_timestamp'] = start
    if gait_df.loc[gait_df.index[-1], 'end_timestamp'] > end:
        gait_df.loc[gait_df.index[-1], 'end_timestamp'] = end

    # Create gait mask
    epoch_ts = list(pd.date_range(start, end, freq='1S', inclusive='both'))

    gait_mask = np.zeros(len(epoch_ts), dtype=int)
    start_stamp = epoch_ts[0]

    # Use nimbalwear/gait.py for the correct format
    try:
        for row in gait_df.itertuples():
            start = np.ceil((row.start_timestamp - start_stamp).total_seconds())
            stop = np.floor((row.end_timestamp - start_stamp).total_seconds())
            gait_mask[int(start):int(stop)] = 1

    # Posture GS format
    except (KeyError, AttributeError):
        for row in gait_df.itertuples():
            start = (row.Start - start_stamp).total_seconds()
            stop = (row.Stop - start_stamp).total_seconds()
            gait_mask[int(start):int(stop)] = 1

    return gait_mask


def _filter_acc_data(x, y, z):
    """Filter accelerometer data to obtain gravity and body movement components

    Args:
        x: time array of the x-axis acceleration values
        y: time array of the y-axis acceleration values
        z: time array of the z-axis acceleration values

    Returns:
        - list of gravity components of each axis of accelerometer data
        - list of body movement components of each axis of accelerometer data
    """
    # median filter to remove high-frequency noise spikes
    denoised_x = signal.medfilt(x)
    denoised_y = signal.medfilt(y)
    denoised_z = signal.medfilt(z)

    # low pass elliptic filter to obtain gravity component
    sos = signal.ellip(3, 0.01, 100, 0.25, 'low', output='sos')
    grav_x = signal.sosfilt(sos, denoised_x)
    grav_y = signal.sosfilt(sos, denoised_y)
    grav_z = signal.sosfilt(sos, denoised_z)

    # subtract gravity component from signal to obtain body movement component
    bm_x = denoised_x - grav_x
    bm_y = denoised_y - grav_y
    bm_z = denoised_z - grav_z

    return [grav_x, grav_y, grav_z], [bm_x, bm_y, bm_z]


def _get_angles(x, y, z):
    """Get angles between each axis using accelerometer data

    Args:
        x: time array of the x-axis acceleration values
        y: time array of the y-axis acceleration values
        z: time array of the z-axis acceleration values

    Returns:
        angle_x: time array of x-axis angles
        angle_y: time array of y-axis angles
        angle_z: time array of z-axis angles
    """
    magnitude = np.sqrt(np.square(np.array([x, y, z])).sum(axis=0))
    angle_x = np.arccos(x / magnitude) * 180 / np.pi
    angle_y = np.arccos(y / magnitude) * 180 / np.pi
    angle_z = np.arccos(z / magnitude) * 180 / np.pi

    return angle_x, angle_y, angle_z


def _get_transitions(x, y, z, gait_mask, th=0.04):
    """Classify static movement time as a transition based on a threshold

    Args:
        x: time array of the x-axis acceleration values
        y: time array of the y-axis acceleration values
        z: time array of the z-axis acceleration values
        gait_mask: time array of 1s and 0s indicating gait activity
        th: threshold for classifying point as a transition (default 0.04)

    Returns:
        time array of 1s and 0s indicating transition
    """
    # get jerk of acceleration data (3rd derivative of position)
    x_jerk = np.diff(x, prepend=x[0])
    y_jerk = np.diff(y, prepend=y[0])
    z_jerk = np.diff(z, prepend=z[0])

    # apply threshold th to the jerks
    x_threshed = np.array([True if x > th else False for x in abs(x_jerk)])
    y_threshed = np.array([True if y > th else False for y in abs(y_jerk)])
    z_threshed = np.array([True if z > th else False for z in abs(z_jerk)])

    # don't include a point as transition if there is gait activity
    return (x_threshed | y_threshed | z_threshed) & \
           np.array([not gait for gait in gait_mask])


def create_posture_df_template(data_df, gait_mask, tran_type='jerk', tran_thresh=0.04):
    """Classify static movement time points as transition based on a threshold. This posture dataframe is then used for
     input into the wrist/ankle/chest/thigh posture classifiers.

    Args:
        data_df: dataframe with timestamps and 3 axes of accelerometer data;
                 must have columns: timestamp, ant_acc, up_acc, left_acc;
                 can be created using prep_accelerometer_df
        gait_mask: time array of 1s and 0s indicating gait activity
        tran_type: optional; string indicating type of transition to use.
                   Options are 'jerk' and 'ang_vel'
        tran_thresh: optional; float or integer indicating the threshold to
                     define a period as a transition

    Returns:
        posture_df: dataframe of timestamps, preliminary postures, angle data,
                 gait mask, and transitions
    """
    # obtain the various components of the posture dataframe
    grav_data, bm_data = _filter_acc_data(
        data_df['ant_acc'], data_df['up_acc'], data_df['left_acc'])
    ant_ang, up_ang, left_ang = _get_angles(*grav_data)

    # find transition periods
    if tran_type == 'jerk':
        transitions = _get_transitions(*bm_data, gait_mask, tran_thresh)
    elif tran_type == 'ang_vel':
        transitions = _get_transitions(ant_ang, up_ang, left_ang, gait_mask, tran_thresh)
    else:
        raise Exception(f"{tran_type} is invalid. Currently accepted options "
                        "are: 'jerk' and 'ang_vel'.")

    # create the posture dataframe
    posture_df = pd.DataFrame(
        {'timestamp': data_df['timestamp'],
         'posture': 'other',
         'ant_ang': ant_ang, 'up_ang': up_ang, 'left_ang': left_ang,
         'gait': gait_mask, 'transition': transitions})

    return posture_df


def classify_ankle_posture(posture_df):
    """Determine posture based on ankle angles from each axis.

    Posture can be: sitstand, sit, horizontal, or other.
    Angle interpretations for posture classification:
        0 degrees: positive axis is pointing upwards (against gravity)
        90 degrees: axis is perpendicular to gravity
        180 degrees: positive axis is pointing downwards (with gravity)

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    # sit/stand: static with ankle tilt 0-25 degrees
    posture_df.loc[(posture_df['up_ang'] < 25), 'posture'] = "sitstand"

    # sit: static with ankle tilt 25-45 degrees
    posture_df.loc[(25 <= posture_df['up_ang']) &
                (posture_df['up_ang'] < 70), 'posture'] = "sit"

    # horizontal: static with shank tilt 45-135 degrees
    posture_df.loc[(70 <= posture_df['up_ang']) &
                (posture_df['up_ang'] <= 135), 'posture'] = "horizontal"

    return posture_df


def classify_chest_posture(posture_df):
    """Determine posture based on chest angles from each axis.

    Posture can be: sitstand, reclined, prone, supine, leftside, rightside, or
    other. Angle interpretations for posture classification:
        0 degrees: positive axis is pointing upwards (against gravity)
        90 degrees: axis is perpendicular to gravity
        180 degrees: positive axis is pointing downwards (with gravity)

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    # sit/stand: static with up axis upwards
    posture_df.loc[(posture_df['up_ang'] < 45), 'posture'] = "sitstand"

    # prone: static with anterior axis downwards, left axis horizontal
    posture_df.loc[(135 <= posture_df['ant_ang']) &
                (45 <= posture_df['left_ang']) & (posture_df['left_ang'] <= 135),
                'posture'] = "prone"

    # supine: static with anterior axis upwards, left & up axes horizontal
    posture_df.loc[(posture_df['ant_ang'] <= 45) &
                (70 <= posture_df['up_ang']) & (45 <= posture_df['left_ang']) &
                (posture_df['left_ang'] <= 135), 'posture'] = "supine"

    # reclined: static with anterior axis upwards, left axis horizontal,
    # up axis above horizontal
    posture_df.loc[(posture_df['ant_ang'] <= 70) &
                (45 <= posture_df['up_ang']) & (posture_df['up_ang'] < 70) &
                (45 <= posture_df['left_ang']) & (posture_df['left_ang'] <= 135),
                'posture'] = "reclined"

    # left side: static with left axis downwards, up axis horizontal
    posture_df.loc[(45 <= posture_df['up_ang']) & (posture_df['up_ang'] <= 135) &
                (135 <= posture_df['left_ang']), 'posture'] = "leftside"

    # right side: static with left axis upwards, up axis horizontal
    posture_df.loc[(45 <= posture_df['up_ang']) & (posture_df['up_ang'] <= 135) &
                (posture_df['left_ang'] < 45), 'posture'] = "rightside"

    return posture_df


def classify_thigh_posture(posture_df):
    """Determine posture based on thigh angles from each axis.

    Posture can be: stand, sitlay, or other.
    Angle interpretations for posture classification:
        0 degrees: positive axis is pointing upwards (against gravity)
        90 degrees: axis is perpendicular to gravity
        180 degrees: positive axis is pointing downwards (with gravity)

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    # stand: thigh tilt 0-45 degrees
    posture_df.loc[(posture_df['up_ang'] < 45), 'posture'] = "stand"

    # sit/lay: static with thigh tilt 45-135 degrees
    posture_df.loc[(45 <= posture_df['up_ang']) &
                (posture_df['up_ang'] < 135), 'posture'] = "sitlay"

    return posture_df


def classify_wrist_position(posture_df):
    """Determine wrist position based on wrist angles from each axis.

    Position can be: up, down, supine, prone, and side.
    Angle interpretations for position classification:
        0 degrees: positive axis is pointing upwards (against gravity)
        90 degrees: axis is perpendicular to gravity
        180 degrees: positive axis is pointing downwards (with gravity)

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    # down: up axis upwards
    posture_df.loc[(posture_df['up_ang'] < 45), 'posture'] = "down"

    # up: up axis downwards
    posture_df.loc[(135 <= posture_df['up_ang']), 'posture'] = "up"

    # prone: left axis upwards
    posture_df.loc[(posture_df['left_ang'] < 45), 'posture'] = "palm_down"

    # supine: left axis downwards
    posture_df.loc[(135 <= posture_df['left_ang']), 'posture'] = "palm_up"

    # thumb_down: up and left axis horizontal, anterior axis down
    posture_df.loc[(45 <= posture_df['up_ang']) & (posture_df['up_ang'] < 135) &
                (45 <= posture_df['left_ang']) & (posture_df['left_ang'] < 135) &
                (135 <= (posture_df['ant_ang'])), 'posture'] = "thumb_down"

    # thumb_up: up and left axis horizontal, anterior axis up
    posture_df.loc[(45 <= posture_df['up_ang']) & (posture_df['up_ang'] < 135) &
                (45 <= posture_df['left_ang']) & (posture_df['left_ang'] < 135) &
                (posture_df['ant_ang'] < 45), 'posture'] = "thumb_up"

    return posture_df


def _crop_posture_dfs(posture_df_dict):
    """
    crops all dfs in the posture_df_dict to the same size by cropping to the latest start and
    earliest end within all posture dataframes
    """
    new_start = max([df['timestamp'].values[0] for df in posture_df_dict.values()])  # Latest start time of any df
    new_end = min([df['timestamp'].values[-1] for df in posture_df_dict.values()])  # Earliest end time of any df

    for k in posture_df_dict.keys():
        df = posture_df_dict[k]
        posture_df_dict[k] = df.loc[(df['timestamp'] >= new_start) & (df['timestamp'] <= new_end)]
        posture_df_dict[k].reset_index(inplace = True)
    return posture_df_dict

def _clean_transitions(transitions, gap_fill=5):
    """Removes single transitions and fills adjacent transition periods
       separated by a maximum of gap of size gap_fill

    Args:
        transitions: time array of 1s and 0s indicating transition
        gap_fill: maximum gap between transition periods to fill transition
                  (default 5)

    Returns:
        transitions: updated time array of 1s and 0s indicating transition
    """
    # removes first and/or last transition if they are alone
    if transitions[0] and not transitions[1]:
        transitions[0] = False
    if transitions[-1] and not transitions[-2]:
        transitions[-1] = False

    # removes any transition that is alone and fills transition gaps
    for i in range(len(transitions)-2):
        if not transitions[i+1] and transitions[i] and \
                i <= len(transitions)-1-gap_fill:
            for j in range(i+1+gap_fill, i-1, -1):
                if transitions[j]:
                    transitions[i:j] = [True for _ in range(i, j)]
                    break
        elif transitions[i+1] and not transitions[i] and not transitions[i+2]:
            transitions[i+1] = False

    return transitions


def combine_posture_dfs(posture_df_dict, tran_combo=None, tran_gap_fill=5):
    """Use available posture dataframes to create a summary posture dataframe

    Args:
        posture_df_dict: dictionary with keys of the wearable name and values of
                         dataframes of timestamps, postures, and transitions
        tran_combo: list of booleans indicating what wearable transitions
                    should be combined for the overall transition mask
        tran_gap_fill: maximum gap between transition periods to fill as
                       transition (default 5)

    Returns:
        posture_df: dataframe of timestamps, combined postures, and combined
                 transitions

    Notes:
        - Must have a chest posture dataframe
        - Does not use wrist data
    """
    print("Combining posture dataframes...", end=" ")
    # Crop dfs to uniform length
    new_posture_df_dict = posture_df_dict.copy()
    new_posture_df_dict = _crop_posture_dfs(new_posture_df_dict)

    # combine transitions based on tran_combination
    if tran_combo is None:
        trans = np.all([df['transition'] for df in new_posture_df_dict.values()], axis=0)
    else:
        if len(tran_combo) != len(new_posture_df_dict.values()):
            raise Exception("Length of tran_combination must match number of "
                            "posture dataframes passed in.")
        else:
            trans = np.all(
                [df['transition'] for i, df in enumerate(new_posture_df_dict.values())
                 if tran_combo[i]], axis=0)

    # clean transitions
    clean_trans = _clean_transitions(trans, tran_gap_fill)

    # create new dataframe, initially setting posture as chest posture
    chest_post = new_posture_df_dict['chest']
    posture_df = pd.DataFrame({'timestamp': chest_post['timestamp'],
                            'posture': chest_post['posture'],
                            'transition': clean_trans,
                            'gait': chest_post['gait']})

    if 'ankle' in new_posture_df_dict:
        ankle_post = new_posture_df_dict['ankle']

        # chest sit/stand + ankle sit = sit
        posture_df.loc[(chest_post['posture'] == 'sitstand') &
                    (ankle_post['posture'] == 'sit'), 'posture'] = "sit"

        # chest sit/stand + ankle horizontal = sitting reclined
        posture_df.loc[(chest_post['posture'] == 'sitstand') &
                    (ankle_post['posture'] == 'horizontal'),
                    'posture'] = "reclined"

        # chest laying + ankle sit or sitstand = other
        posture_df.loc[((chest_post['posture'] == 'prone') |
                     (chest_post['posture'] == 'supine') |
                     (chest_post['posture'] == 'rightside') |
                     (chest_post['posture'] == 'leftside')) &
                    ((ankle_post['posture'] == 'sit') |
                     (ankle_post['posture'] == 'sitstand')),
                    'posture'] = 'other'

    if 'thigh' in new_posture_df_dict:
        thigh_post = new_posture_df_dict['thigh']

        # chest sit/stand + thigh stand = stand
        posture_df.loc[(posture_df['posture'] != 'reclined') &
                    (chest_post['posture'] == 'sitstand') &
                    (thigh_post['posture'] == 'stand'), 'posture'] = "stand"

        # chest sit/stand + thigh sitlay = sit
        posture_df.loc[(posture_df['posture'] != 'reclined') &
                    (chest_post['posture'] == 'sitstand') &
                    (thigh_post['posture'] == 'sitlay'), 'posture'] = "sit"

    # overwrite postures within gait as stand
    posture_df.loc[(posture_df['gait']) == 1, 'posture'] = "stand"

    # overwrite postures within transition as transition
    posture_df.loc[(posture_df['transition']), 'posture'] = "transition"

    print("Completed.")
    return posture_df

def summarize_posture_df(posture_df):
    """Compresses consecutive rows of the same posture (with/without labels
       into one row of the posture dataframe.
    """
    print("Compressing dataframe...", end=" ")
    df = posture_df.copy()
    durations = []
    first_diff_idx = 0
    last_idx = df.shape[0] - 1
    prev_start = df.iloc[0]['timestamp']
    prev_post = df.iloc[0]['posture']


    for row in df.itertuples():
        compress_bool = (row.posture != prev_post)
        if compress_bool:
            if (first_diff_idx + 1) != row.Index:
                df.drop([i for i in range(first_diff_idx + 1, row.Index)],
                        inplace=True)

            durations.append((row.timestamp - prev_start).total_seconds())

            first_diff_idx = row.Index
            prev_start = row.timestamp
            prev_post = row.posture

            if row.Index == last_idx:
                durations.append(1)

        elif row.Index == last_idx:
            df.drop([i for i in range(first_diff_idx + 1, last_idx + 1)],
                    inplace=True)
            durations.append(
                (row.timestamp + td(seconds=1) - prev_start).total_seconds())
    df.reset_index(drop=True, inplace=True)
    df.insert(1, 'duration', durations)
    end_timestamps = [row.timestamp + td(seconds=row.duration)
                      for row in df.itertuples()]
    df.rename(columns={'timestamp': 'start_timestamp'})
    df.insert(1, 'end_timestamp', end_timestamps)
    df.drop(['transition', 'gait'], axis=1, inplace=True)

    print("Completed.")
    return df

def posture(gait_df, wrist_x = None,wrist_y = None,wrist_z = None, wrist_freq = None, wrist_start = None,
            ankle_x = None,ankle_y = None,ankle_z = None, ankle_freq = None, ankle_start = None,
            chest_x = None, chest_y = None,chest_z = None, chest_freq = None, chest_start = None,
            thigh_x = None, thigh_y = None, thigh_z = None, thigh_freq = None, thigh_start = None,
            tran_combo=None, tran_gap_fill=5):
    """
    Wrapper function to get gait for an individual with multiple sensors

    Args:
        gait_df: This is a summary dataframe that is created in nimbalwear.gait. It should have the following
                 columns ['start', 'end', 'number_steps', 'start_timestamp', 'end_timestamp']
        tran_combo: list of booleans indicating what wearable transitions
                    should be combined for the overall transition mask
        tran_gap_fill: maximum gap between transition periods to fill as
                       transition (default 5)

    Returns:
        posture_dfs_dict: a dictionary containing a dataframe of postures, timestamps and axis values for each location
        summary_posture_dfs_dict: Start and stop times for each posture for the dataframes in posture_dfs_dict
        final_posture_df: Dataframe which combines the different locations to create an optimal posture prediction
        summary_final_posture_df: Start and stop times for each posture in final_posture_df

    Notes:
        For the final_posture_df to work, it must have a chest accelerometer provided.
    """
    wearables_dict = {}
    if not [x for x in  (wrist_x, wrist_y, wrist_z, wrist_freq, wrist_start) if x is None]:
        wearables_dict['wrist'] = [wrist_x, wrist_y, wrist_z, wrist_freq, wrist_start]
    if not [x for x in  (ankle_x, ankle_y, ankle_z, ankle_freq, ankle_start) if x is None]:
        wearables_dict['ankle'] = [ankle_x, ankle_y, ankle_z, ankle_freq, ankle_start]
    if not [x for x in  (chest_x, chest_y, chest_z, chest_freq, chest_start) if x is None]:
        wearables_dict['chest'] = [chest_x, chest_y, chest_z, chest_freq, chest_start]
    if not [x for x in  (thigh_x, thigh_y, thigh_z, thigh_freq, thigh_start) if x is None]:
        wearables_dict['thigh'] = [thigh_x, thigh_y, thigh_z, thigh_freq, thigh_start]

    posture_dfs_dict = {}
    summary_posture_dfs_dict = {}
    for bodypart, vals in wearables_dict.items():
        x = vals[0]
        y = vals[1]
        z = vals[2]
        freq = vals[3]
        start = vals[4]

        prepped_df = prep_accelerometer_df(x, y, z, sample_rate=freq, body_location=bodypart, start_datetime=start)
        end = prepped_df['timestamp'].values[-1]

        gait_mask = get_gait_mask(gait_df, start, end)

        posture_df_temp = create_posture_df_template(prepped_df, gait_mask)

        if bodypart == 'ankle':
            posture_dfs_dict['ankle'] = classify_ankle_posture(posture_df_temp)
            summary_posture_dfs_dict['ankle'] = summarize_posture_df(posture_dfs_dict['ankle'])
        elif bodypart == 'chest':
            posture_dfs_dict['chest'] = classify_chest_posture(posture_df_temp)
            summary_posture_dfs_dict['chest'] = summarize_posture_df(posture_dfs_dict['chest'])
        elif bodypart == 'thigh':
            posture_dfs_dict['thigh'] = classify_thigh_posture(posture_df_temp)
            summary_posture_dfs_dict['thigh'] = summarize_posture_df(posture_dfs_dict['thigh'])
        elif bodypart == 'wrist':
            posture_dfs_dict['wrist'] = classify_wrist_position(posture_df_temp)
            summary_posture_dfs_dict['wrist'] = summarize_posture_df(posture_dfs_dict['wrist'])
    final_posture_df = combine_posture_dfs(posture_dfs_dict, tran_combo=tran_combo, tran_gap_fill=tran_gap_fill)
    summary_final_posture_df = summarize_posture_df(final_posture_df)


    return posture_dfs_dict, summary_posture_dfs_dict, final_posture_df, summary_final_posture_df