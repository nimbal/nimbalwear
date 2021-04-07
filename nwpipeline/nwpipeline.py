
import os
import datetime as dt
from pathlib import Path

from tqdm import tqdm
import pandas as pd

from nwpipeline import nwdata

class NWPipeline:

    def __init__(self, study_dir):

        # folder constants
        self.study_dir = os.path.abspath(study_dir)
        self.raw_data_dir = os.path.join(self.study_dir, 'raw_data')
        self.standard_device_dir = os.path.join(self.study_dir, 'processed_data/standard_device_edf')
        self.cropped_device_dir = os.path.join(self.study_dir, 'processed_data/cropped_device_edf')
        self.sensor_dir = os.path.join(self.study_dir, 'processed_data/sensor_edf')

        # file path constants
        self.device_list_path = os.path.join(self.study_dir, 'device_list.csv')
        self.nonwear_csv = os.path.join(self.study_dir, 'analyzed_data/nonwear/standard_nonwear_times.csv')

        # read device list
        self.device_list = pd.read_csv(os.path.join(self.study_dir, self.device_list_path), dtype=str, )

    def process_coll(self, subject_id, coll_id, quiet=False):

        # read all data files

        import_switch = {'GNAC': lambda: device_data.import_gnac(device_raw_path, correct_drift=True, quiet=quiet),
                         'BITF': lambda: device_data.import_edf(device_raw_path),
                         'NONW': lambda: device_data.import_nonw(device_raw_path, quiet=quiet)}

        sensors_switch = {'GNAC': ['ACCELEROMETER', 'TEMPERATURE', 'LIGHT', 'BUTTON'],
                          'BITF': ['ACCELEROMETER', 'ECG'],
                          'NONW': ['PLSOX']}

        sensor_channels_switch = {'GNAC': [[0, 1, 2], [3], [4], [5]],
                                  'BITF': [[1, 2, 3], [0]],
                                  'NONW': [[0, 1]]}

        # search for files from subject visit
        subject_device_list_df = self.device_list.loc[(self.device_list['subject_id'] == subject_id) &
                                                    (self.device_list['coll_id'] == coll_id)]

        # read in all data files for one subject
        for index, row in tqdm(subject_device_list_df.iterrows(), desc='Processing devices'):

            # get info from device list
            device_type = row['device_type']
            device_location = row['device_location']
            device_file_name = row['file_name']

            # evaluate device type cases
            import_func = import_switch.get(device_type, lambda: 'Invalid')
            sensors = sensors_switch.get(device_type, lambda: 'Invalid')
            sensor_channels = sensor_channels_switch.get(device_type, lambda: 'Invalid')

            # create all file path variables
            device_file_base = os.path.splitext(device_file_name)[0]

            device_raw_path = os.path.join(self.raw_data_dir, device_type, device_file_name)

            device_edf_name = '.'.join([device_file_base, 'edf'])

            sensor_edf_names = ['.'.join(['_'.join([device_file_base, sensor]), 'edf']) for sensor in sensors]

            standard_device_path = os.path.join(self.standard_device_dir, device_type, device_edf_name)
            cropped_device_path = os.path.join(self.cropped_device_dir, device_type, device_edf_name)
            sensor_paths = [os.path.join(self.sensor_dir, device_type, sensors[sen], sensor_edf_names[sen])
                            for sen in range(len(sensors))]

            # check file and folder structure
            if not quiet:
                print("Checking file and folder structure ...")

            # check that raw data file exists
            if not os.path.isfile(device_raw_path):
                print(f"WARNING: {device_raw_path} does not exist.\n")
                return

            # check that all folders exist for data output files
            Path(os.path.dirname(standard_device_path)).mkdir(parents=True, exist_ok=True)
            Path(os.path.dirname(cropped_device_path)).mkdir(parents=True, exist_ok=True)
            for sensor_path in sensor_paths:
                Path(os.path.dirname(sensor_path)).mkdir(parents=True, exist_ok=True)

            # READ DEVICE DATA
            if not quiet:
                print("Reading device data...")

            # import data to device data object

            device_data = nwdata.NWData()
            import_func()
            device_data.deidentify()

            # TODO: check header against device list info

            # write device data as edf
            if not quiet:
                print("Writing device .edf file ...")
            device_data.export_edf(file_path=standard_device_path)

            # CROP FINAL NONWEAR FROM DEVICE
            if not quiet:
                print("Cropping device .edf file ...")

            # read nonwear csv file
            nonwear_df = pd.read_csv(self.nonwear_csv, dtype=str)
            nonwear_df['start_time'] = pd.to_datetime(nonwear_df['start_time'], format='%Y-%m-%d %H:%M')
            nonwear_df['end_time'] = pd.to_datetime(nonwear_df['end_time'], format='%Y-%m-%d %H:%M')

            # get last device nonwear period
            last_nonwear = nonwear_df.loc[(nonwear_df['subject_id'] == subject_id) &
                                          (nonwear_df['coll_id'] == coll_id) &
                                          (nonwear_df['device_type'] == device_type) &
                                          (nonwear_df['device_location'] == device_location)][-1:]

            # get time info from device data
            start_time = device_data.header['startdate']
            duration = dt.timedelta(seconds=len(device_data.signals[0]) / device_data.signal_headers[0]['sample_rate'])
            end_time = start_time + duration

            nonwear_duration = dt.timedelta(minutes=0)
            nonwear_time_to_eof = dt.timedelta(minutes=21)

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

            # write as cropped device data as edf
            device_data.export_edf(file_path=cropped_device_path)

            # SEPARATE DEVICE EDF TO SENSOR EDFS
            if not quiet:
                print("Writing sensor .edf files...")

            for sen in tqdm(range(len(sensor_paths)), desc="Separating sensors"):

                sen_path = sensor_paths[sen]
                sen_channels = sensor_channels[sen]

                device_data.export_edf(file_path=sen_path, sig_nums_out=sen_channels)

