import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
from tqdm import tqdm
import pandas as pd

import nwfiles.file.EDF as edf

device_path_1 = '/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/cropped_sensor_edf'
device_path_2 = '/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/standard_sensor_edf'

sensor_paths = ['ACCELEROMETER',
                'TEMPERATURE',
                'LIGHT',
                'BUTTON']

csv_name = 'compare.csv'

for sensor_path in tqdm(sensor_paths):

    sensor_path_1 = os.path.join(device_path_1, sensor_path)
    sensor_path_2 = os.path.join(device_path_2, sensor_path)
    csv_path = os.path.join(sensor_path_1, csv_name)

    file_list = [f for f in os.listdir(sensor_path_1)
                 if f.lower().endswith('.edf') and not f.startswith('.')]
    file_list.sort()

    subjects = []
    device_locations = []
    file_1_starts = []
    file_1_ends = []
    file_2_starts = []
    file_2_ends = []
    start_diffs = []
    end_diffs = []

    for file_name in tqdm(file_list):

        file_base = os.path.splitext(file_name)[0].replace('_' + sensor_path, '')

        # read file 1
        header, signal_headers = edf.read_header(os.path.join(sensor_path_1, file_name))
        subject = header['patientcode']
        device_location = header['recording_additional']
        edf_1_start = header['startdate']
        edf_1_end = edf_1_start + header['duration']

        # read file 2
        header, signal_headers = edf.read_header(os.path.join(sensor_path_2, file_name))
        edf_2_start = header['startdate']
        edf_2_end = edf_2_start + header['duration']

        subjects.append(subject)
        device_locations.append(device_location)
        file_1_starts.append(edf_1_start)
        file_1_ends.append(edf_1_end)
        file_2_starts.append(edf_2_start)
        file_2_ends.append(edf_2_end)
        start_diffs.append(edf_2_start - edf_1_start)
        end_diffs.append(edf_2_end - edf_1_end)

        # last_nonwear = nonwear_df.loc[(nonwear_df['ID'] == int(subject[-4:])) &
        #                               (nonwear_df['location'] == device_location[:2])][-1:]

    comparison_df = pd.DataFrame({'subject': subjects,
                                  'device_location': device_locations,
                                  'file_1_start': file_1_starts,
                                  'file_2_start': file_2_starts,
                                  'file_1_end': file_1_ends,
                                  'file_2_end': file_2_ends,
                                  'start_diff': start_diffs,
                                  'end_diff': end_diffs})

    comparison_df.to_csv(csv_path, index=False)
