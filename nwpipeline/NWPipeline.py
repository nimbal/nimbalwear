import os
import datetime as dt
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import nwdata

# TODO: fix tqdm progress bar locations

class NWPipeline:

    def __init__(self, study_dir):

        # initialize folder structure
        self.study_dir = os.path.abspath(study_dir)

        self.dirs = {
            'study': '',
            'meta': 'meta',
            'logs': 'meta/logs',
            'raw': 'raw',
            'processed': 'processed',
            'standard_device_edf': 'processed/standard_device_edf',
            'cropped_device_edf': 'processed/cropped_device_edf',
            'sensor_edf': 'processed/sensor_edf',
            'analyzed': 'analyzed',
            'nonwear': 'analyzed/nonwear',
            'activity': 'analyzed/activity',
            'gait': 'analyzed/gait',
            'sleep': 'analyzed/sleep'}

        self.dirs = {key: os.path.join(self.study_dir, value) for key, value in self.dirs.items()}

        # pipeline data files
        self.device_list_path = os.path.join(self.dirs['meta'], 'device_list.csv')

        # TODO: check for required files (raw data, device_list)

        # read device list
        self.device_list = pd.read_csv(self.device_list_path, dtype=str).fillna('')

        # TODO: initialize folder structure
        for key, value in self.dirs.items():
            Path(value).mkdir(parents=True, exist_ok=True)

    def run(self, subject_ids=None, coll_ids=None, overwrite_header=False, quiet=False):

        # TODO: if no subject_ids or coll_ids, then do all

        for subject_id in tqdm(subject_ids, desc="Processing subjects", leave=True):

            for coll_id in tqdm(coll_ids, desc="Processing collections", leave=False):

                self.coll_proc(subject_id=subject_id, coll_id=coll_id, overwrite_header=overwrite_header, quiet=quiet)

    def coll_proc(self, subject_id, coll_id, overwrite_header=False, quiet=False):

        # TODO: create collection class with device, device_list, nonwear, attributes

        # get devices for this collection from device_list
        coll_device_list_df = self.device_list.loc[(self.device_list['subject_id'] == subject_id) &
                                                   (self.device_list['coll_id'] == coll_id)]
        coll_device_list_df.reset_index(inplace=True, drop=True)

        # create collection class and process
        coll = NWCollection(subject_id=subject_id, coll_id=coll_id, device_list=coll_device_list_df, dirs=self.dirs)
        coll.process(overwrite_header=overwrite_header, min_crop_duration=3, max_crop_time_to_eof=20, quiet=quiet)

    def get_subject_ids(self):

        subject_ids = self.device_list['subject_id'].unique()
        subject_ids.sort()

        return subject_ids

    def get_coll_ids(self):

        coll_ids = self.device_list['coll_id'].unique()
        coll_ids.sort()

        return coll_ids


class NWCollection:

    sensors_switch = {'GNAC': ['ACCELEROMETER', 'TEMPERATURE', 'LIGHT', 'BUTTON'],
                      'BITF': ['ACCELEROMETER', 'ECG'],
                      'NONW': ['PLSOX']}

    sensor_channels_switch = {'GNAC': [[0, 1, 2], [3], [4], [5]],
                              'BITF': [[1, 2, 3], [0]],
                              'NONW': [[0, 1]]}

    devices = []
    nonwear_times = None

    def __init__(self, subject_id, coll_id, device_list, dirs):

        self.subject_id = subject_id
        self.coll_id = coll_id
        self.device_list = device_list
        self.dirs = dirs

    def process(self, overwrite_header=False, min_crop_duration=1, max_crop_time_to_eof=20, quiet=False):

        # read data from all devices in collection
        self.read(overwrite_header=overwrite_header, save=True, quiet=quiet)

        # synchronize devices

        # process nonwear for all devices
        self.nonwear()

        # crop final nonwear
        self.crop(save=True, quiet=quiet, min_duration=min_crop_duration, max_time_to_eof=max_crop_time_to_eof)

        # save sensor edf files
        self.save_sensors()

        # process activity levels

        # process gait

        # process sleep

    def read(self, overwrite_header=False, save=False, rename_file=False, quiet=False):

        import_switch = {'GNAC': lambda: device_data.import_gnac(device_raw_path, correct_drift=True, quiet=quiet),
                         'BITF': lambda: device_data.import_bitf(device_raw_path),
                         'NONW': lambda: device_data.import_nonw(device_raw_path, quiet=quiet)}

        self.devices = []

        # read in all data files for one subject
        for index, row in tqdm(self.device_list.iterrows(), total=self.device_list.shape[0], leave=False,
                               desc='Reading all device data'):

            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_id = row['device_id']
            device_location = row['device_location']
            device_file_name = row['file_name']

            device_raw_path = os.path.join(self.dirs['raw'], device_type, device_file_name)

            # check that raw data file exists
            if not os.path.isfile(device_raw_path):
                print(f"WARNING: {device_raw_path} does not exist.\n")
                self.devices.append(None)
                continue

            # TODO: log entry if file doesn't exist

            import_func = import_switch.get(device_type, lambda: 'Invalid')

            # import data to device data object
            device_data = nwdata.NWData()
            import_func()
            device_data.deidentify()

            # check header against device list info
            header_comp = {'subject_id': (device_data.header['patientcode'] == subject_id),
                           'coll_id': (device_data.header['patient_additional'] == coll_id),
                           'device_type': (device_data.header['equipment'].split('_')[0] == device_type),
                           'device_id': (device_data.header['equipment'].split('_')[1] == device_id
                                         if len(device_data.header['equipment'].split('_')) > 1 else False),
                           'device_location': (device_data.header['recording_additional'] == device_location)}

            # TODO: log entry if checks fail and what was overwritten

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
                standard_device_path = os.path.join(self.dirs['standard_device_edf'], device_type, device_edf_name)

                # check that all folders exist for data output files
                Path(os.path.dirname(standard_device_path)).mkdir(parents=True, exist_ok=True)

                # write device data as edf
                device_data.export_edf(file_path=standard_device_path)

            self.devices.append(device_data)

        return True

    def nonwear(self):

        # process nonwear for all devices
        nonwear_csv = os.path.join(self.dirs['nonwear'], 'standard_nonwear_times.csv')

        # read nonwear csv file
        self.nonwear_times = pd.read_csv(nonwear_csv, dtype=str)
        self.nonwear_times['start_time'] = pd.to_datetime(self.nonwear_times['start_time'], format='%Y-%m-%d %H:%M')
        self.nonwear_times['end_time'] = pd.to_datetime(self.nonwear_times['end_time'], format='%Y-%m-%d %H:%M')

        return True

    def crop(self, save=False, quiet=False, min_duration=1, max_time_to_eof=20):

        # crop final nonwear from all device data
        for index, row in tqdm(self.device_list.iterrows(), total=self.device_list.shape[0], leave=False,
                               desc='Cropping final nonwear'):

            if self.devices[index] is None:
                continue

            # get info from device list
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_file_name = row['file_name']

            # get last device nonwear period
            last_nonwear = self.nonwear_times.loc[(self.nonwear_times['subject_id'] == subject_id) &
                                                  (self.nonwear_times['coll_id'] == coll_id) &
                                                  (self.nonwear_times['device_type'] == device_type) &
                                                  (self.nonwear_times['device_location'] == device_location)][-1:]

            # get time info from device data
            start_time = self.devices[index].header['startdate']
            duration = dt.timedelta(
                seconds=len(self.devices[index].signals[0]) / self.devices[index].signal_headers[0]['sample_rate'])
            end_time = start_time + duration

            nonwear_duration = dt.timedelta(minutes=0)
            nonwear_time_to_eof = dt.timedelta(minutes=max_time_to_eof + 1)

            if not last_nonwear.empty:
                # get duration and time to end of file of last nonwear
                nonwear_duration = last_nonwear['end_time'].item() - last_nonwear['start_time'].item()
                nonwear_time_to_eof = end_time - last_nonwear['end_time'].item()

            # only crop if last nonwear ends within 20 minutes of end of file
            is_cropped = ((nonwear_duration >= dt.timedelta(minutes=min_duration)) &
                          (nonwear_time_to_eof <= dt.timedelta(minutes=max_time_to_eof)))

            # set new file end time to which to crop
            new_start_time = start_time
            new_end_time = last_nonwear['start_time'].item() if is_cropped else end_time

            self.devices[index].crop(new_start_time, new_end_time)

            if save:

                if not quiet:
                    print("Saving cropped device .edf file ...")

                # create all file path variables
                device_file_base = os.path.splitext(device_file_name)[0]
                device_edf_name = '.'.join([device_file_base, 'edf'])

                cropped_device_path = os.path.join(self.dirs['cropped_device_edf'], device_type, device_edf_name)

                # check that all folders exist for data output files
                Path(os.path.dirname(cropped_device_path)).mkdir(parents=True, exist_ok=True)

                # write cropped device data as edf
                self.devices[index].export_edf(file_path=cropped_device_path)

        return True

    def save_sensors(self):

        for index, row in tqdm(self.device_list.iterrows(), total=self.device_list.shape[0], leave=False,
                               desc='Saving sensor edfs'):

            if self.devices[index] is None:
                continue

            # get info from device list
            device_type = row['device_type']
            device_file_name = row['file_name']

            # TODO: check that all device types in list are valid before running

            # evaluate device type cases
            sensors = self.sensors_switch.get(device_type, lambda: 'Invalid')
            sensor_channels = self.sensor_channels_switch.get(device_type, lambda: 'Invalid')

            # create all file path variables
            device_file_base = os.path.splitext(device_file_name)[0]
            sensor_edf_names = ['.'.join(['_'.join([device_file_base, sensor]), 'edf']) for sensor in sensors]

            sensor_paths = [os.path.join(self.dirs['sensor_edf'], device_type, sensors[sen], sensor_edf_names[sen])
                            for sen in range(len(sensors))]

            # check that all folders exist for data output files
            for sensor_path in sensor_paths:
                Path(os.path.dirname(sensor_path)).mkdir(parents=True, exist_ok=True)

            for sen in tqdm(range(len(sensor_paths)), leave=False, desc="Separating sensors"):

                sen_path = sensor_paths[sen]
                sen_channels = sensor_channels[sen]

                self.devices[index].export_edf(file_path=sen_path, sig_nums_out=sen_channels)

        return True
