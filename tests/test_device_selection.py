import pandas as pd

study_code = 'test_ReMiNDD'
subject_id = '1027'
coll_id = '01'

sensors_switch = {'GNAC': ['ACCELEROMETER', 'TEMPERATURE', 'LIGHT', 'BUTTON'],
                  'AXV6': ['GYROSCOPE', 'ACCELEROMETER', 'LIGHT', 'TEMPERATURE'],
                  'BITF': ['ACCELEROMETER', 'ECG'],
                  'NONW': ['PLSOX']}

device_locations = {'left_ankle': ['LA', 'LEFTANKLE', 'LANKLE'],
                    'left_wrist': ['LW', 'LEFTWRIST', 'LWRIST'],
                    'right_wrist': ['RW', 'RIGHTWRIST', 'RWRIST'],
                    'right_ankle': ['RA', 'RIGHTANKLE', 'RANKLE']}

dominant = False

device_info_path = '/Volumes/KIT_DATA/test_ReMiNDD/meta/devices.csv'
subject_info_path = '/Volumes/KIT_DATA/test_ReMiNDD/meta/subjects.csv'


device_info = pd.read_csv(device_info_path, dtype=str).fillna('')
subject_info = pd.read_csv(subject_info_path, dtype=str).fillna('')

device_info = device_info.loc[(device_info['study_code'] == study_code) &
                                           (device_info['subject_id'] == subject_id) &
                                           (device_info['coll_id'] == coll_id)]

subject_info = subject_info.loc[(subject_info['study_code'] == study_code) &
                                        (subject_info['subject_id'] == subject_id)]

subject_info = subject_info.iloc[0].to_dict() if subject_info.shape[0] > 0 else {}

print(device_info)
print(subject_info)

# select devices

# activity
device_info = device_info.loc[(device_info['study_code'] == study_code) &
                                           (device_info['subject_id'] == subject_id) &
                                           (device_info['coll_id'] == coll_id)]

activity_device_types = ['GNAC', 'AXV6']
activity_locations = device_locations['right_wrist'] + device_locations['left_wrist']

activity_device_index = device_info.loc[(device_info['device_type'].isin(activity_device_types)) &
                                     (device_info['device_location'].isin(activity_locations))].index.values.tolist()

print(activity_device_index)

# if multiple wrist accelerometers
if len(activity_device_index) > 1:

    if subject_info['dominant_hand']:

        # select dominant or non-dominant based on argument
        if dominant:
            wrist = 'right_wrist' if subject_info['dominant_hand'] == 'right' else 'left_wrist'
        else:
            wrist = 'left_wrist' if subject_info['dominant_hand'] == 'right' else 'right_wrist'

        activity_locations = device_locations[wrist]
        activity_device_index = device_info.loc[(device_info['device_type'].isin(activity_device_types)) &
                                             (device_info['device_location'].isin(activity_locations))].index.values.tolist()

        # if still multiple take first one,
        if len(activity_device_index) > 1:
            activity_device_index = [activity_device_index[0]]

        # if none, go back and take first one from list of all
        elif len(activity_device_index) < 1:
            activity_locations = device_locations['right_wrist'] + device_locations['left_wrist']
            activity_device_index = device_info.loc[(device_info['device_type'].isin(activity_device_types)) &
                                                 (device_info['device_location'].isin(activity_locations))].index.values.tolist()
            activity_device_index = [activity_device_index[0]]

    else:
        activity_device_index = [activity_device_index[0]]

# if only one device determine, if it is dominant
elif len(activity_device_index) == 1:

    # if dominant hand info is available we will determine dominance
    if subject_info['dominant_hand']:
        dominant_wrist = subject_info['dominant_hand'] + '_wrist'
        dominant = device_info.loc[activity_device_index]['device_location'].item() in device_locations[dominant_wrist]



# if no wrist accelerometers
#elif len(activity_device_index) < 1:





