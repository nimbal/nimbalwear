import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
import datetime
from tqdm import tqdm
import pandas as pd

import nwfiles.file.EDF as edf

cropped_device_path = '/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/cropped_sensor_edf'
uncropped_device_path = '/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/standard_sensor_edf'

sensor_paths = ['ACCELEROMETER',
                'TEMPERATURE',
                'LIGHT',
                'BUTTON']

nonwear_path = ('/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/standard_nonwear_times/' +
               'GNAC_standard_nonwear_times.csv')

csv_name = 'compare.csv'

nonwear_df = pd.read_csv(nonwear_path)
nonwear_df['start_time'] = pd.to_datetime(nonwear_df['start_time'], format='%Y-%m-%d %H:%M')
nonwear_df['end_time'] = pd.to_datetime(nonwear_df['end_time'], format='%Y-%m-%d %H:%M')

for sensor_path in tqdm(sensor_paths):

    cropped_path = os.path.join(cropped_device_path, sensor_path)
    uncropped_path = os.path.join(uncropped_device_path, sensor_path)
    csv_path = os.path.join(cropped_path, csv_name)

    file_list = [f for f in os.listdir(cropped_path)
                 if f.lower().endswith('.edf') and not f.startswith('.')]
    file_list.sort()

    subjects = []
    device_locations = []
    cr_starts = []
    cr_ends = []
    un_starts = []
    un_ends = []
    start_crops = []
    end_crops = []
    last_nw_starts = []
    last_nw_diffs = []

    for file_name in tqdm(file_list):

        file_base = os.path.splitext(file_name)[0].replace('_' + sensor_path, '')

        # read cropped file
        header, signal_headers = edf.read_header(os.path.join(cropped_path, file_name))
        subject = header['patientcode']
        device_location = header['recording_additional']
        cr_start = header['startdate']
        cr_end = cr_start + header['duration']

        # read uncropped file
        header, signal_headers = edf.read_header(os.path.join(uncropped_path, file_name))
        un_start = header['startdate']
        un_end = un_start + header['duration']

        subjects.append(subject)
        device_locations.append(device_location)
        cr_starts.append(cr_start)
        cr_ends.append(cr_end)
        un_starts.append(un_start)
        un_ends.append(un_end)
        start_crops.append(cr_start - un_start)
        end_crops.append(un_end - cr_end)

        # last_nonwear = nonwear_df.loc[(nonwear_df['ID'] == int(subject[-4:])) &
        #                               (nonwear_df['location'] == device_location[:2])][-1:]

        # get last device nonwear period
        last_nonwear = nonwear_df.loc[(nonwear_df['file_name'] == file_base)][-1:]

        if not last_nonwear.empty:
            last_nw_start = last_nonwear['start_time'].item()
            last_nw_diff = cr_end - last_nw_start
        else:
            last_nw_start = 'NA'
            last_nw_diff = 'NA'

        last_nw_starts.append(last_nw_start)
        last_nw_diffs.append(last_nw_diff)

    comparison_df = pd.DataFrame({'subject': subjects,
                                  'device_location': device_locations,
                                  'cr_start': cr_starts,
                                  'un_start': un_starts,
                                  'cr_end': cr_ends,
                                  'un_end': un_ends,
                                  'last_nw_start': last_nw_starts,
                                  'start_crop': start_crops,
                                  'end_crop': end_crops,
                                  'last_nw_dif': last_nw_diffs})

    comparison_df.to_csv(csv_path, index=False)
