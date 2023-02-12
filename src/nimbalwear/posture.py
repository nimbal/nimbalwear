import pandas as pd
import numpy as np
from scipy import signal
from datetime import timedelta as td


def prep_accelerometer_df(freq, body_location, start_time,
                          wrist_palmar=None, wrist_proximal=None, wrist_thumb=None,
                          ankle_anterior=None, ankle_proximal=None, ankle_lateral=None,
                          chest_superior=None, chest_left=None, chest_anterior=None,
                          thigh_anterior=None, thigh_proximal=None, thigh_lateral=None):
    """
    Creates the proper dataframe format for an individual accelerometer for the remainder of the posture analysis
    """
    # Get data
    if body_location == 'chest':
        ts = pd.date_range(start_time, periods=len(chest_superior),
                           freq=(str(1 / freq) + 'S'))
        df = pd.DataFrame({'timestamp': ts, 'superior_acc': chest_superior,
                           'left_acc': chest_left, 'anterior_acc': chest_anterior})
    elif body_location == 'ankle':
        ts = pd.date_range(start_time, periods=len(ankle_proximal),
                           freq=(str(1 / freq) + 'S'))
        df = pd.DataFrame({'timestamp': ts, 'anterior_acc': ankle_anterior,
                           'proximal_acc': ankle_proximal, 'lateral_acc': ankle_lateral})
    elif body_location == 'thigh':
        ts = pd.date_range(start_time, periods=len(thigh_proximal),
                           freq=(str(1 / freq) + 'S'))
        df = pd.DataFrame({'timestamp': ts, 'anterior_acc': thigh_anterior,
                           'proximal_acc': thigh_proximal, 'lateral_acc': thigh_lateral})
    elif body_location == 'wrist':
        ts = pd.date_range(start_time, periods=len(wrist_proximal),
                           freq=(str(1 / freq) + 'S'))
        df = pd.DataFrame({'timestamp': ts, 'palmar_acc': wrist_palmar,
                           'proximal_acc': wrist_proximal, 'thumb_acc': wrist_thumb})

    return df.resample(
                f'{1}S', on='timestamp').mean().reset_index(col_fill='timestamp')  # Resample to 1s intervals using mean


def get_gait_mask(gait_df, start_time, end_time):
    """
    creates a list of 0's (no gait) and 1's (gait) with each value representing 1 second.
    """
    # Read in gait_object

    gait_df['start_time'] = pd.DatetimeIndex(gait_df['start_time']).floor('S')
    gait_df['end_time'] = pd.DatetimeIndex(gait_df['end_time']).ceil('S')

    # Crop gait df to start and end times
    if gait_df.loc[gait_df.index[0], 'start_time'] < start_time:
        gait_df.loc[gait_df.index[0], 'start_time'] = start_time
    if gait_df.loc[gait_df.index[-1], 'end_time'] > end_time:
        gait_df.loc[gait_df.index[-1], 'end_time'] = end_time

    # Create gait mask
    epoch_ts = list(pd.date_range(start_time, end_time, freq='1S', inclusive='both'))
    gait_mask = np.zeros(len(epoch_ts), dtype=int)
    start_timestamp = epoch_ts[0]  # Change start_time to pandas timestamp format

    for row in gait_df.itertuples():
        start = np.ceil((row.start_time - start_timestamp).total_seconds())
        stop = np.floor((row.end_time - start_timestamp).total_seconds())
        gait_mask[int(start):int(stop)] = 1

    return gait_mask


def _filter_acc_data(x,y,z):
    """Filter accelerometer data to obtain gravity and body movement components

    Args:
        x: time array of the single-axis acceleration values
        y: time array of the single-axis acceleration values
        z: time array of the single-axis acceleration values

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


def _get_angles(grav_x, grav_y, grav_z):
    """Get angles between each axis using the gravity components of the accelerometer data created in _filter_acc_data

    Args:
        grav_x: time array of the gravity component of a single-axis of an accelerometer
        grav_y: time array of the gravity component of a single-axis of an accelerometer
        grav_z: time array of the gravity component of a single-axis of an accelerometer

    Returns:
        angle_x: time array of angles for a single-axis
        angle_y: time array of angles for a single-axis
        angle_z: time array of angles for a single-axis

    Notes:
        All angles are calculated with 0 being pointed towards gravity. For example, someone standing in the anatomical
        position would have their anterior facing 90 degrees and superior at 180 degrees
    """
    magnitude = np.sqrt(np.square(np.array([grav_x, grav_y, grav_z])).sum(axis=0))
    angle_x = np.arccos(grav_x / magnitude) * 180 / np.pi
    angle_y = np.arccos(grav_y / magnitude) * 180 / np.pi
    angle_z = np.arccos(grav_z / magnitude) * 180 / np.pi

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
    x_threshed = np.abs(x_jerk) > th
    y_threshed = np.abs(y_jerk) > th
    z_threshed = np.abs(z_jerk) > th

    # don't include a point as transition if there is gait activity
    if gait_mask is not None:
        return (x_threshed | y_threshed | z_threshed) & \
           np.array([not gait for gait in gait_mask])
    else:
         return(x_threshed | y_threshed | z_threshed)


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
    temp_df = data_df.drop(columns ='timestamp')
    a, b, c = zip(*temp_df.values)
    grav_data, bm_data = _filter_acc_data(a, b, c)
    angs = _get_angles(*grav_data)

    # find transition periods
    if tran_type == 'jerk':
        transitions = _get_transitions(*bm_data, gait_mask, tran_thresh)
    elif tran_type == 'ang_vel':
        transitions = _get_transitions(*angs, gait_mask, tran_thresh)
    else:
        raise Exception(f"{tran_type} is invalid. Currently accepted options "
                        "are: 'jerk' and 'ang_vel'.")

    # create the posture dataframe
    col_names = [col.split('_')[0] for col in temp_df.columns]
    ang_names = [col+'_ang' for col in col_names]
    df_dict = {'timestamp':data_df['timestamp'], 'posture':'other'}
    for n, ang_name in enumerate(ang_names):
        df_dict[ang_name] = angs[n]
    df_dict['gait'] = gait_mask
    df_dict['transition'] = transitions
    posture_df = pd.DataFrame(df_dict)

    return posture_df


def classify_ankle_posture(anterior, proximal, lateral, gait_df, freq, start_time):
    """Determine posture based on ankle angles from each axis. TODO: Add classification details to external documentation
    Posture can be: sitstand, sit, horizontal, or other.

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    prepped_df = prep_accelerometer_df(ankle_anterior=anterior, ankle_proximal=proximal, ankle_lateral=lateral,
                                       freq=freq, body_location='ankle', start_time=start_time)
    end = prepped_df['timestamp'].values[-1]

    if gait_df is not None:
        gait_mask = get_gait_mask(gait_df, start_time, end)
    else:
        gait_mask = None

    posture_df = create_posture_df_template(prepped_df, gait_mask)

    # sit/stand: static with ankle tilt 0-25 degrees
    posture_df.loc[(posture_df['proximal_ang'] < 25), 'posture'] = "sitstand"

    # sit: static with ankle tilt 25-45 degrees
    posture_df.loc[(25 <= posture_df['proximal_ang']) &
                   (posture_df['proximal_ang'] < 70), 'posture'] = "sit"

    # horizontal: static with shank tilt 45-135 degrees
    posture_df.loc[(70 <= posture_df['proximal_ang']) &
                   (posture_df['proximal_ang'] <= 135), 'posture'] = "horizontal"

    return posture_df


def classify_chest_posture(superior, left, anterior, gait_df, freq, start_time):
    """Determine posture based on chest angles from each axis. TODO: Add classification details to external documentation

    Posture can be: sitstand, reclined, prone, supine, leftside, rightside, or
    other.

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    prepped_df = prep_accelerometer_df(chest_superior=superior, chest_left=left, chest_anterior=anterior, freq=freq,
                                       body_location='chest', start_time=start_time)
    end = prepped_df['timestamp'].values[-1]

    if gait_df is not None:
        gait_mask = get_gait_mask(gait_df, start_time, end)
    else:
        gait_mask = None

    posture_df = create_posture_df_template(prepped_df, gait_mask)

    # sit/stand: static with up axis upwards
    posture_df.loc[(posture_df['superior_ang'] < 45), 'posture'] = "sitstand"

    # prone: static with anterior axis downwards, left axis horizontal
    posture_df.loc[(135 <= posture_df['anterior_ang']) &
                   (45 <= posture_df['left_ang']) & (posture_df['left_ang'] <= 135),
                   'posture'] = "prone"

    # supine: static with anterior axis upwards, left & up axes horizontal
    posture_df.loc[(posture_df['anterior_ang'] <= 45) &
                   (70 <= posture_df['superior_ang']) & (45 <= posture_df['left_ang']) &
                   (posture_df['left_ang'] <= 135), 'posture'] = "supine"

    # reclined: static with anterior axis upwards, left axis horizontal,
    # up axis above horizontal
    posture_df.loc[(posture_df['anterior_ang'] <= 70) &
                   (45 <= posture_df['superior_ang']) & (posture_df['superior_ang'] < 70) &
                   (45 <= posture_df['left_ang']) & (posture_df['left_ang'] <= 135),
                   'posture'] = "reclined"

    # left side: static with left axis downwards, up axis horizontal
    posture_df.loc[(45 <= posture_df['superior_ang']) & (posture_df['superior_ang'] <= 135) &
                   (135 <= posture_df['left_ang']), 'posture'] = "leftside"

    # right side: static with left axis upwards, up axis horizontal
    posture_df.loc[(45 <= posture_df['superior_ang']) & (posture_df['superior_ang'] <= 135) &
                   (posture_df['left_ang'] < 45), 'posture'] = "rightside"

    return posture_df


def classify_thigh_posture(anterior, proximal, lateral, gait_df, freq, start_time):
    """Determine posture based on thigh angles from each axis. TODO: Add classification details to external documentation
    Posture can be: stand, sitlay, or other.

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    prepped_df = prep_accelerometer_df(thigh_anterior=anterior, thigh_proximal=proximal, thigh_lateral=lateral,
                                       freq=freq, body_location='thigh', start_time=start_time)
    end = prepped_df['timestamp'].values[-1]

    if gait_df is not None:
        gait_mask = get_gait_mask(gait_df, start_time, end)
    else:
        gait_mask = None

    posture_df = create_posture_df_template(prepped_df, gait_mask)

    # stand: thigh tilt 0-45 degrees
    posture_df.loc[(posture_df['proximal_ang'] < 45), 'posture'] = "stand"

    # sit/lay: static with thigh tilt 45-135 degrees
    posture_df.loc[(45 <= posture_df['proximal_ang']) &
                   (posture_df['proximal_ang'] < 135), 'posture'] = "sitlay"

    return posture_df


def classify_wrist_position(palmar, proximal, thumb, gait_df, freq, start_time):
    """Determine posture based on wrist angles from each axis. TODO: Add classification details to external documentation
    Position can be: up, down, supine, prone, and side.

    Args:
        posture_df: dataframe of timestamps, angle data, and postures created in create_posture_df_template

    Returns:
        posture_df: copy of the inputted posture_df with and updated postures
    """
    prepped_df = prep_accelerometer_df(wrist_palmar=palmar, wrist_proximal=proximal, wrist_thumb=thumb, freq=freq,
                                       body_location='wrist', start_time=start_time)
    end = prepped_df['timestamp'].values[-1]

    if gait_df is not None:
        gait_mask = get_gait_mask(gait_df, start_time, end)
    else:
        gait_mask = None

    posture_df = create_posture_df_template(prepped_df, gait_mask)

    # down: up axis upwards
    posture_df.loc[(posture_df['proximal_ang'] < 45), 'posture'] = "down"

    # up: up axis downwards
    posture_df.loc[(135 <= posture_df['proximal_ang']), 'posture'] = "up"

    # prone: left axis upwards
    posture_df.loc[(posture_df['thumb_ang'] < 45), 'posture'] = "palm_down"

    # supine: left axis downwards
    posture_df.loc[(135 <= posture_df['thumb_ang']), 'posture'] = "palm_up"

    # thumb_down: up and left axis horizontal, anterior axis down
    posture_df.loc[(45 <= posture_df['proximal_ang']) & (posture_df['proximal_ang'] < 135) &
                   (45 <= posture_df['thumb_ang']) & (posture_df['thumb_ang'] < 135) &
                   (135 <= (posture_df['palmar_ang'])), 'posture'] = "thumb_down"

    # thumb_up: up and left axis horizontal, anterior axis up
    posture_df.loc[(45 <= posture_df['proximal_ang']) & (posture_df['proximal_ang'] < 135) &
                   (45 <= posture_df['thumb_ang']) & (posture_df['thumb_ang'] < 135) &
                   (posture_df['palmar_ang'] < 45), 'posture'] = "thumb_up"

    return posture_df


def _crop_posture_dfs(posture_df_dict):
    """
    crops all dfs in the posture_df_dict to the same size by cropping to the latest start and
    earliest end within all posture dataframes

    Args:
        posture_df_dict: dictionary with keys of the wearable name and values of dataframes of timestamps, postures,
                         and transitions
    """
    # Get start and stop times
    new_start = max([df['timestamp'].values[0] for df in posture_df_dict.values()])  # Latest start time of any df
    new_end = min([df['timestamp'].values[-1] for df in posture_df_dict.values()])  # Earliest end time of any df

    # Crop Dfs
    for k in posture_df_dict.keys():
        df = posture_df_dict[k]
        posture_df_dict[k] = df.loc[(df['timestamp'] >= new_start) & (df['timestamp'] <= new_end)]
        posture_df_dict[k].reset_index(inplace=True)
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
    for i in range(len(transitions) - 2):
        if not transitions[i + 1] and transitions[i] and \
                i <= len(transitions) - 1 - gap_fill:
            for j in range(i + 1 + gap_fill, i - 1, -1):
                if transitions[j]:
                    transitions[i:j] = [True for _ in range(i, j)]
                    break
        elif transitions[i + 1] and not transitions[i] and not transitions[i + 2]:
            transitions[i + 1] = False

    return transitions


def combine_posture_dfs(posture_df_dict, tran_combo=None, tran_gap_fill=5):
    """Use available posture dataframes to create a combined posture dataframe. This DataFrame will default to the chest
    posture and will use additional posture dataframes to improve the classification.

    TODO: Add specific details about how posture is determined in external documentation

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
        - Columns in the returned dataframe are ['timestamp', 'posture', 'transition', 'gait']
    """
    print("Combining posture dataframes...", end=" ")
    # Crop dfs to uniform length
    new_posture_df_dict = posture_df_dict.copy()
    if 'wrist' in new_posture_df_dict.keys():
        new_posture_df_dict.pop('wrist')
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
        # If chest is reclined or if chest
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

    # Remove transitions that have the same posture bordering it on both sides
    summary_df = summarize_posture_df(posture_df)
    for row in summary_df.loc[summary_df['posture'] == 'transition'].itertuples():
        if row.Index == 0 or row.Index == len(summary_df):
            continue

        if summary_df.iloc[row.Index - 1]['posture'] == summary_df.iloc[row.Index + 1]['posture']:
            actual_posture = summary_df.iloc[row.Index - 1]['posture']
            posture_df.loc[(posture_df['timestamp'] >= row.start_time) & (
                        posture_df['timestamp'] <= row.end_time), 'posture'] = actual_posture

    print("Completed.")
    return posture_df


def summarize_posture_df(posture_df):
    """Compresses posture_df into just the distinct successive postures. The columns for this dataframe are
    ['start_time', 'end_time', 'duration', 'posture']
    """
    print("Compressing dataframe...", end=" ")
    df = posture_df.copy()
    vals_changed = df["posture"].shift(-1) != df["posture"]
    summary_df = df.copy()[vals_changed]
    durs = summary_df['timestamp'] - summary_df['timestamp'].shift(1)
    durs.iloc[0] = summary_df['timestamp'].iloc[0] - df['timestamp'].iloc[0]
    summary_df['duration'] = durs.dt.total_seconds().values

    start_times = df.iloc[(summary_df['timestamp'].shift(1).index + 1)[:-1].insert(0, 0)]['timestamp']
    summary_df['start_time'] = start_times.values
    summary_df.rename(columns={'timestamp': 'end_time'}, inplace=True)
    summary_df = summary_df[['start_time', 'end_time', 'duration', 'posture']]
    summary_df.reset_index(drop=True, inplace=True)
    print("Completed.")
    return summary_df


def posture_detect(gait_df = None, wrist_palmar=None, wrist_proximal=None, wrist_thumb=None, wrist_freq=None, wrist_start=None,
                   ankle_anterior=None, ankle_proximal=None, ankle_lateral=None, ankle_freq=None, ankle_start=None,
                   chest_superior=None, chest_left=None, chest_anterior=None, chest_freq=None, chest_start=None,
                   thigh_anterior=None, thigh_proximal=None, thigh_lateral=None, thigh_freq=None, thigh_start=None,
                   tran_combo=None, tran_gap_fill=5):
    """
    Wrapper function to get gait for an individual with multiple sensors

    Args:
        gait_df: This is a summary dataframe that is created in nimbalwear.gait. It should have the following
                 columns ['start', 'end', 'number_steps', 'start_time', 'end_time']
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
    if not [x for x in (wrist_palmar, wrist_proximal, wrist_thumb, wrist_freq, wrist_start) if x is None]:
        wearables_dict['wrist'] = {'palmar': wrist_palmar, 'proximal': wrist_proximal, 'thumb': wrist_thumb,
                                   'freq': wrist_freq, 'start': wrist_start}
    if not [x for x in (ankle_anterior, ankle_proximal, ankle_lateral, ankle_freq, ankle_start) if x is None]:
        wearables_dict['ankle'] = {'anterior': ankle_anterior, 'proximal': ankle_proximal, 'lateral': ankle_lateral,
                                   'freq': ankle_freq, 'start': ankle_start}
    if not [x for x in (chest_superior, chest_left, chest_anterior, chest_freq, chest_start) if x is None]:
        wearables_dict['chest'] = {'superior': chest_superior, 'left': chest_left, 'anterior': chest_anterior,
                                   'freq': chest_freq, 'start': chest_start}
    if not [x for x in (thigh_proximal, thigh_freq, thigh_start) if x is None]:
        wearables_dict['thigh'] = {'anterior': thigh_anterior, 'proximal': thigh_proximal, 'lateral': thigh_lateral,
                                   'freq': thigh_freq, 'start': thigh_start}

    posture_dfs_dict = {}
    summary_posture_dfs_dict = {}
    for bodypart in wearables_dict:
        bodypart_dict = wearables_dict[bodypart]

        if bodypart == 'ankle':
            posture_dfs_dict['ankle'] = classify_ankle_posture(anterior=bodypart_dict['anterior'],
                                                               proximal=bodypart_dict['proximal'],
                                                               lateral=bodypart_dict['lateral'], gait_df=gait_df,
                                                               freq=bodypart_dict['freq'],
                                                               start_time=bodypart_dict['start'])
            summary_posture_dfs_dict['ankle'] = summarize_posture_df(posture_dfs_dict['ankle'])
        elif bodypart == 'chest':
            posture_dfs_dict['chest'] = classify_chest_posture(superior=bodypart_dict['superior'],
                                                               left=bodypart_dict['left'],
                                                               anterior=bodypart_dict['anterior'], gait_df=gait_df,
                                                               freq=bodypart_dict['freq'],
                                                               start_time=bodypart_dict['start'])
            summary_posture_dfs_dict['chest'] = summarize_posture_df(posture_dfs_dict['chest'])
        elif bodypart == 'thigh':
            posture_dfs_dict['thigh'] = classify_thigh_posture(anterior=bodypart_dict['anterior'],
                                                               proximal=bodypart_dict['proximal'],
                                                               lateral=bodypart_dict['lateral'], gait_df=gait_df,
                                                               freq=bodypart_dict['freq'],
                                                               start_time=bodypart_dict['start'])
            summary_posture_dfs_dict['thigh'] = summarize_posture_df(posture_dfs_dict['thigh'])
        elif bodypart == 'wrist':
            posture_dfs_dict['wrist'] = classify_wrist_position(palmar=bodypart_dict['palmar'],
                                                                proximal=bodypart_dict['proximal'],
                                                                thumb=bodypart_dict['thumb'], gait_df=gait_df,
                                                                freq=bodypart_dict['freq'],
                                                                start_time=bodypart_dict['start'])
            summary_posture_dfs_dict['wrist'] = summarize_posture_df(posture_dfs_dict['wrist'])
    final_posture_df = combine_posture_dfs(posture_dfs_dict, tran_combo=tran_combo, tran_gap_fill=tran_gap_fill)
    summary_final_posture_df = summarize_posture_df(final_posture_df)

    return posture_dfs_dict, summary_posture_dfs_dict, final_posture_df, summary_final_posture_df
