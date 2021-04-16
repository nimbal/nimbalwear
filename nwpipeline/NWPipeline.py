import os
import datetime as dt
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import nwdata


class NWPipeline:

    def __init__(self, study_dir):

        # TODO: more dynamic structure (dict?, list?)

        # folder constants
        self.study_dir = os.path.abspath(study_dir)
        self.raw_data_dir = os.path.join(self.study_dir, 'raw_data')
        self.processed_data_dir = os.path.join(self.study_dir, 'processed_data')
        self.analyzed_dir = os.path.join(self.study_dir, 'analyzed_data')

        # processed data subdirs
        self.standard_device_dir = os.path.join(self.processed_data_dir, 'standard_device_edf')
        self.cropped_device_dir = os.path.join(self.processed_data_dir, 'cropped_device_edf')
        self.sensor_dir = os.path.join(self.processed_data_dir, 'sensor_edf')

        # analyzed data subdirs
        self.nonwear_dir = os.path.join(self.analyzed_dir, 'nonwear')
        self.activity_dir = os.path.join(self.analyzed_dir, 'activity')
        self.gait_dir = os.path.join(self.analyzed_dir, 'gait')
        self.sleep_dir = os.path.join(self.analyzed_dir, 'sleep')

        # pipeline data files
        self.device_list_path = os.path.join(self.study_dir, 'device_list.csv')

        # nonwear files
        self.nonwear_csv = os.path.join(self.nonwear_dir, 'standard_nonwear_times.csv')

        # TODO: initialize folder structure

        # read device list
        self.device_list = pd.read_csv(self.device_list_path, dtype=str).fillna('')

    def get_subject_ids(self):

        subject_ids = self.device_list['subject_id'].unique()
        subject_ids.sort()

        return subject_ids

    def run(self, subject_ids = None, coll_ids = None, overwrite_header=False, quiet=False):

        for subject_id in subject_ids:

            for coll_id in coll_ids:

                self.coll_proc(subject_id=subject_id, coll_id=coll_id, overwrite_header=overwrite_header, quiet=quiet)

    def coll_proc(self, subject_id, coll_id, overwrite_header=False, quiet=False):

        # read data from all devices in collection
        devices = self.coll_read(subject_id, coll_id, overwrite_header=overwrite_header, save=True, quiet=quiet)

        # synchronize devices

        # process nonwear for all devices

        # crop final nonwear
        devices = self.coll_crop(subject_id, coll_id, devices, save=True, quiet=quiet)

        # save sensor edf files
        self.coll_sens(subject_id, coll_id, devices)

        # process activity levels

        # process gait

        # process sleep

    def coll_read(self, subject_id, coll_id, overwrite_header=False, save=False, rename_file=False, quiet=False):

        import_switch = {'GNAC': lambda: device_data.import_gnac(device_raw_path, correct_drift=True, quiet=quiet),
                         'BITF': lambda: device_data.import_bitf(device_raw_path),
                         'NONW': lambda: device_data.import_nonw(device_raw_path, quiet=quiet)}

        # get devices for subject visit from device_list
        subject_device_list_df = self.device_list.loc[(self.device_list['subject_id'] == subject_id) &
                                                      (self.device_list['coll_id'] == coll_id)]

        devices = {}

        # read in all data files for one subject
        for index, row in tqdm(subject_device_list_df.iterrows(), desc='Reading all device data'):

            device_type = row['device_type']
            device_id = row['device_id']
            device_location = row['device_location']
            device_file_name = row['file_name']

            device_raw_path = os.path.join(self.raw_data_dir, device_type, device_file_name)

            # check that raw data file exists
            if not os.path.isfile(device_raw_path):
                print(f"WARNING: {device_raw_path} does not exist.\n")
                devices[index] = None
                return

            import_func = import_switch.get(device_type, lambda: 'Invalid')

            # import data to device data object
            device_data = nwdata.NWData()
            import_func()
            device_data.deidentify()

            # TODO: change device codes in nwdata import functions to match pipeline by default?

            # check header against device list info
            header_comp = {'subject_id': (device_data.header['patientcode'] == subject_id),
                           'coll_id': (device_data.header['patient_additional'] == coll_id),
                           'device_type': (device_data.header['equipment'].split('_')[0] == device_type),
                           'device_id': (device_data.header['equipment'].split('_')[1] == device_id
                                         if len(device_data.header['equipment'].split('_')) > 1 else False),
                           'device_location': (device_data.header['recording_additional'] == device_location)}

            # TODO: provide feedback (table?, log?) of these checks and what was overwritten

            if overwrite_header:

                device_data.header['patientcode'] = subject_id
                device_data.header['patient_additional'] = coll_id
                device_data.header['equipment'] = '_'.join([device_type, device_id])
                device_data.header['recording_additional'] = device_location

            if save:

                if not quiet:
                    print("Saving device .edf file ...")

                # TODO: option to rename files

                # create all file path variables
                device_file_base = os.path.splitext(device_file_name)[0]
                device_edf_name = '.'.join([device_file_base, 'edf'])
                standard_device_path = os.path.join(self.standard_device_dir, device_type, device_edf_name)

                # check that all folders exist for data output files
                Path(os.path.dirname(standard_device_path)).mkdir(parents=True, exist_ok=True)

                # write device data as edf
                device_data.export_edf(file_path=standard_device_path)

            devices[index] = device_data

        return devices

    def coll_crop(self, subject_id, coll_id, devices, save=False, quiet=False):

        # get devices for subject visit from device_list - REMOVE LATER
        subject_device_list_df = self.device_list.loc[(self.device_list['subject_id'] == subject_id) &
                                                      (self.device_list['coll_id'] == coll_id)]

        # crop final nonwear from all device data
        for index, row in tqdm(subject_device_list_df.iterrows(), desc='Cropping final nonwear'):

            # get info from device list
            device_type = row['device_type']
            device_location = row['device_location']
            device_file_name = row['file_name']

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
            start_time = devices[index].header['startdate']
            duration = dt.timedelta(
                seconds=len(devices[index].signals[0]) / devices[index].signal_headers[0]['sample_rate'])
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

            devices[index].crop(new_start_time, new_end_time)

            if save:

                if not quiet:
                    print("Saving cropped device .edf file ...")

                # create all file path variables
                device_file_base = os.path.splitext(device_file_name)[0]
                device_edf_name = '.'.join([device_file_base, 'edf'])

                cropped_device_path = os.path.join(self.cropped_device_dir, device_type, device_edf_name)

                # check that all folders exist for data output files
                Path(os.path.dirname(cropped_device_path)).mkdir(parents=True, exist_ok=True)

                # write cropped device data as edf
                devices[index].export_edf(file_path=cropped_device_path)

            return devices

    def coll_sens(self, subject_id, coll_id, devices):

        sensors_switch = {'GNAC': ['ACCELEROMETER', 'TEMPERATURE', 'LIGHT', 'BUTTON'],
                          'BITF': ['ACCELEROMETER', 'ECG'],
                          'NONW': ['PLSOX']}

        sensor_channels_switch = {'GNAC': [[0, 1, 2], [3], [4], [5]],
                                  'BITF': [[1, 2, 3], [0]],
                                  'NONW': [[0, 1]]}

        # get devices for subject visit from device_list - REMOVE LATER
        subject_device_list_df = self.device_list.loc[(self.device_list['subject_id'] == subject_id) &
                                                      (self.device_list['coll_id'] == coll_id)]

        for index, row in tqdm(subject_device_list_df.iterrows(), desc='Saving sensor edfs'):

            # get info from device list
            device_type = row['device_type']
            device_file_name = row['file_name']

            # evaluate device type cases
            sensors = sensors_switch.get(device_type, lambda: 'Invalid')
            sensor_channels = sensor_channels_switch.get(device_type, lambda: 'Invalid')

            # create all file path variables
            device_file_base = os.path.splitext(device_file_name)[0]
            sensor_edf_names = ['.'.join(['_'.join([device_file_base, sensor]), 'edf']) for sensor in sensors]

            sensor_paths = [os.path.join(self.sensor_dir, device_type, sensors[sen], sensor_edf_names[sen])
                            for sen in range(len(sensors))]

            # check that all folders exist for data output files
            for sensor_path in sensor_paths:
                Path(os.path.dirname(sensor_path)).mkdir(parents=True, exist_ok=True)

            for sen in tqdm(range(len(sensor_paths)), desc="Separating sensors"):

                sen_path = sensor_paths[sen]
                sen_channels = sensor_channels[sen]

                devices[index].export_edf(file_path=sen_path, sig_nums_out=sen_channels)
