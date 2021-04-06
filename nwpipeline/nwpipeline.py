import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
import datetime as dt
from pathlib import Path
from fnmatch import fnmatch

from tqdm import tqdm
import pandas as pd

from nwpipeline import nwdata as nw


def process_gnac(study_dir, file_patterns=['*'], nonwear_csv=None, convert_edf=True,
                 separate_sensors=True, crop_nonwear=True, quiet=False):

    bin_dir = study_dir + '/raw_data/GNAC/'
    standard_device_dir = study_dir + '/processed_data/GNAC/standard_device_edf'
    standard_sensor_dir = study_dir + '/processed_data/GNAC/standard_sensor_edf'
    cropped_sensor_dir = study_dir + '/processed_data/GNAC/cropped_sensor_edf'

    file_list = [f for f in os.listdir(bin_dir)
                 if f.endswith('.bin') and not f.startswith('.')
                 and any([fnmatch(f, file_pattern) for file_pattern in file_patterns])]
    file_list.sort()

    for file_name in tqdm(file_list, desc='Processing GNAC files'):

        bin_path = os.path.join(bin_dir, file_name)

        process_single_gnac(bin_path=bin_path, standard_device_dir=standard_device_dir,
                            standard_sensor_dir=standard_sensor_dir, cropped_sensor_dir=cropped_sensor_dir,
                            nonwear_csv=nonwear_csv, convert_edf=convert_edf, separate_sensors=separate_sensors,
                            crop_nonwear=crop_nonwear, quiet=quiet)


def process_single_gnac(bin_path, standard_device_dir, standard_sensor_dir, cropped_sensor_dir, nonwear_csv,
                        convert_edf=True, separate_sensors=True, crop_nonwear=True, quiet=False):

    bin_file_name = os.path.basename(bin_path)
    file_base = os.path.splitext(bin_file_name)[0]

    device_edf_name = '.'.join([file_base, 'edf'])

    sensors = ['ACCELEROMETER',
               'TEMPERATURE',
               'LIGHT',
               'BUTTON']

    sensor_channels = [[0, 1, 2], [3], [4], [5]]

    sensor_edf_names = ['.'.join(['_'.join([file_base, sensor]), 'edf']) for sensor in sensors]

    standard_device_path = os.path.join(standard_device_dir, device_edf_name)
    standard_sensor_paths = [os.path.join(standard_sensor_dir, sensors[sen], sensor_edf_names[sen])
                             for sen in range(len(sensors))]
    cropped_sensor_paths = [os.path.join(cropped_sensor_dir, sensors[sen], sensor_edf_names[sen])
                            for sen in range(len(sensors))]

    if not quiet:
        print("Reading data...")

    device_data = None

    if convert_edf:
        if os.path.isfile(bin_path):
            device_data = nw.nwdata()
            device_data.import_gnac(bin_path, correct_drift=True, quiet=quiet)
            device_data.deidentify()
    elif separate_sensors or crop_nonwear:
        if os.path.isfile(standard_device_path):
            device_data = nw.nwdata()
            device_data.import_edf(standard_device_path)
            device_data.deidentify()

    if device_data is None:
        print("Unable to read data.")
        return

    # CONVERT FROM BIN TO EDF

    if convert_edf:

        Path(standard_device_dir).mkdir(parents=True, exist_ok=True)

        # write as edf
        device_data.export_edf(file_path=standard_device_path)

    # SEPARATE DEVICE EDF TO SENSOR EDFS

    if separate_sensors:

        if not quiet:
            print("Saving sensor .edf files...")

        for sensor in sensors:
            Path(os.path.join(standard_sensor_dir, sensor)).mkdir(parents=True, exist_ok=True)

        for sen in tqdm(range(len(standard_sensor_paths)), desc="Separating sensors"):

            sen_path = standard_sensor_paths[sen]
            sen_channels = sensor_channels[sen]

            device_data.export_edf(file_path=sen_path, sig_nums_out=sen_channels)

    # CROP NONWEAR FROM SENSOR EDFS
    if crop_nonwear:

        if not quiet:
            print("Cropping sensor .edf files ...")

        # crop final nonwear
        nonwear_df = pd.read_csv(nonwear_csv)
        nonwear_df['start_time'] = pd.to_datetime(nonwear_df['start_time'], format='%Y-%m-%d %H:%M')
        nonwear_df['end_time'] = pd.to_datetime(nonwear_df['end_time'], format='%Y-%m-%d %H:%M')

        # get subject_id and device_location from input file
        subject_id = device_data.header['patientcode']
        device_location = device_data.header['recording_additional']
        start_time = device_data.header['startdate']
        duration = dt.timedelta(seconds=len(device_data.signals[0]) / device_data.signal_headers[0]['sample_rate'])
        end_time = start_time + duration

        nonwear_duration = dt.timedelta(minutes=0)
        nonwear_time_to_eof = dt.timedelta(minutes=21)

        # # get last device nonwear period
        # last_nonwear = nonwear_df.loc[(nonwear_df['subject_id'] == int(subject_id)) &
        #                               (nonwear_df['device_location'] == device_location)][-1:]

        # get last device nonwear period
        last_nonwear = nonwear_df.loc[(nonwear_df['file_name'] == file_base)][-1:]

        if not last_nonwear.empty:
            # get duration and time to end of file of last nonwear
            nonwear_duration = last_nonwear['end_time'].item() - last_nonwear['start_time'].item()
            nonwear_time_to_eof = end_time - last_nonwear['end_time'].item()

        # only crop if last nonwear ends within 20 minutes of end of file
        is_cropped = ((nonwear_duration >= dt.timedelta(minutes=3)) &
                      (nonwear_time_to_eof < dt.timedelta(minutes=20)))

        # set new file end time to which to crop
        new_start_time = start_time
        new_end_time = last_nonwear['start_time'].item() if is_cropped else end_time

        device_data.crop(new_start_time, new_end_time)

        for sensor in sensors:
            Path(os.path.join(cropped_sensor_dir, sensor)).mkdir(parents=True, exist_ok=True)

        for sen in tqdm(range(len(cropped_sensor_paths)), desc="Saving cropped sensor files"):

            cropped_sensor_path = cropped_sensor_paths[sen]
            sen_channels = sensor_channels[sen]

            device_data.export_edf(file_path=cropped_sensor_path, sig_nums_out=sen_channels)
