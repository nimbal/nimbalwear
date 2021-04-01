import sys
sys.path.append(r'/Users/kbeyer/repos')

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import pyedflib
from tqdm import tqdm

import nwfiles.file.EDF as edf

device_in_path = '/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_sensor_edf'

device_out_path = '/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/cropped_sensor_edf'

sensor_paths = ['ACCELEROMETER',
                'TEMPERATURE',
                'LIGHT',
                'BUTTON']

nonwear_csv = (r'/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_nonwear_times/' +
               'ReMiNDDNonWearReformatted_GAgoldstandarddataset_04March2021_MinsToEnd.csv')
overwrite = False

nonwear_df = pd.read_csv(nonwear_csv)
nonwear_df['start_time'] = pd.to_datetime(nonwear_df['start_time'], format='%Y-%m-%d %H:%M')
nonwear_df['end_time'] = pd.to_datetime(nonwear_df['end_time'], format='%Y-%m-%d %H:%M')

for sensor_path in tqdm(sensor_paths):

    in_path = os.path.join(device_in_path, sensor_path)
    out_path = os.path.join(device_out_path, sensor_path)

    Path(out_path).mkdir(parents=True, exist_ok=True)

    file_list = [f for f in os.listdir(in_path)
                 if f.lower().endswith('.edf') and not f.startswith('.')]

    if not overwrite:
        file_list = [f for f in file_list if f not in os.listdir(out_path)]

    file_list.sort()

    for file_name in tqdm(file_list):

        in_file = os.path.join(in_path, file_name)
        out_file = os.path.join(out_path, file_name)

        # get subject_id and device_location from input file
        edf_reader = pyedflib.EdfReader(in_file)
        subject_id = edf_reader.getPatientCode()[-4:]
        device_location = edf_reader.getRecordingAdditional()[:2]
        start_time = edf_reader.getStartdatetime()
        duration = dt.timedelta(seconds=edf_reader.getFileDuration())
        end_time = start_time + duration
        edf_reader.close()

        nonwear_duration = dt.timedelta(hours=0)
        nonwear_time_to_eof = dt.timedelta(minutes=21)

        # get last device nonwear period
        last_nonwear = nonwear_df.loc[(nonwear_df['ID'] == int(subject_id)) &
                                      (nonwear_df['location'] == device_location)][-1:]

        if not last_nonwear.empty:
            # get duration and time to end of file of last nonwear
            nonwear_duration = last_nonwear['end_time'].item() - last_nonwear['start_time'].item()
            nonwear_time_to_eof = end_time - last_nonwear['end_time'].item()

        # only crop if last nonwear is at least 2 hours long and ends within 20 minutes of end of file
        is_cropped = ((nonwear_duration >= dt.timedelta(hours=2)) &
                      (nonwear_time_to_eof < dt.timedelta(minutes=20)))

        # set new file end time to which to crop
        new_start_time = start_time
        new_end_time = last_nonwear['start_time'].item() if is_cropped else end_time

        edf.crop(in_file, out_file, new_start_time, new_end_time, overwrite=overwrite)
