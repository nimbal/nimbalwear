import os
import datetime as dt
from pathlib import Path
import logging

from tqdm import tqdm
import pandas as pd
import nwdata
import nwnonwear


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
            'standard_nonwear_times': 'analyzed/nonwear/standard_nonwear_times',
            'activity': 'analyzed/activity',
            'gait': 'analyzed/gait',
            'sleep': 'analyzed/sleep'}

        self.dirs = {key: os.path.join(self.study_dir, value) for key, value in self.dirs.items()}

        # pipeline data files
        self.device_list_path = os.path.join(self.dirs['meta'], 'device_list.csv')

        # TODO: check for required files (raw data, device_list)

        # read device list
        self.device_list = pd.read_csv(self.device_list_path, dtype=str).fillna('')

        # initialize folder structure
        for key, value in self.dirs.items():
            Path(value).mkdir(parents=True, exist_ok=True)

        # TODO: remove further folder checks in methods?? Can't remove where device folder is required because
        # created ad hoc for now

        log_file_path = os.path.join(self.dirs['logs'], "processing.log")
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=log_file_path,
                            level=logging.DEBUG)

    def run(self, subject_ids=None, coll_ids=None, overwrite_header=False, quiet=False, log=True):

        message("\n\n", level='info', display=(not quiet), log=log)
        message("---- Start processing pipeline ----------------------------------------------",
                level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        # if no subject_ids passed then do all
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        # if no coll_ids passed then do all
        if coll_ids is None:
            coll_ids = self.get_coll_ids()

        message(f"Subjects: {subject_ids}", level='info', display=(not quiet), log=log)
        message(f"Collections: {coll_ids}", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        for subject_id in tqdm(subject_ids, desc="Processing subjects", leave=True):

            for coll_id in tqdm(coll_ids, desc="Processing collections", leave=False):

                self.coll_proc(subject_id=subject_id, coll_id=coll_id, overwrite_header=overwrite_header,
                               quiet=quiet, log=log)

        message("---- End ----------------------------------------------\n", level='info', display=(not quiet), log=log)

    def coll_proc(self, subject_id, coll_id, overwrite_header=False, quiet=False, log=True):

        message(f"---- Subject {subject_id}, Collection {coll_id} --------", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        # get devices for this collection from device_list
        coll_device_list_df = self.device_list.loc[(self.device_list['subject_id'] == subject_id) &
                                                   (self.device_list['coll_id'] == coll_id)]
        coll_device_list_df.reset_index(inplace=True, drop=True)

        # construct collection class and process
        coll = NWCollection(subject_id=subject_id, coll_id=coll_id, device_list=coll_device_list_df,
                            dirs=self.dirs)
        coll.process(overwrite_header=overwrite_header, min_crop_duration=3, max_crop_time_to_eof=20,
                     quiet=quiet, log=log)

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
    nonwear_times = pd.DataFrame()

    def __init__(self, subject_id, coll_id, device_list, dirs):

        self.subject_id = subject_id
        self.coll_id = coll_id
        self.device_list = device_list
        self.dirs = dirs

    def process(self, overwrite_header=False, min_crop_duration=1, max_crop_time_to_eof=20, quiet=False, log=True):

        # read data from all devices in collection
        self.read(overwrite_header=overwrite_header, save=True, quiet=quiet, log=log)

        # data integrity ??

        # synchronize devices

        # process nonwear for all devices
        self.nonwear(save=True, quiet=quiet, log=log)

        # crop final nonwear
        self.crop(save=True, min_duration=min_crop_duration, max_time_to_eof=max_crop_time_to_eof, quiet=quiet, log=log)

        # save sensor edf files
        self.save_sensors(quiet=quiet, log=log)

        # process posture

        # process activity levels

        # process gait

        # process sleep

    def read(self, overwrite_header=False, save=False, rename_file=False, quiet=False, log=True):

        message("Reading device data from files...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

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

            message(f"Reading {device_raw_path}", level='info', display=(not quiet), log=log)

            # check that raw data file exists
            if not os.path.isfile(device_raw_path):

                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: {device_raw_path} does not exist", level='warning', display=(not quiet), log=log)
                self.devices.append(None)
                continue

            import_func = import_switch.get(device_type, lambda: 'Invalid')

            # import data to device data object
            device_data = nwdata.NWData()
            import_func()
            device_data.deidentify()

            # check header against device list info
            header_comp = {'subject_id': [(device_data.header['patientcode'] == subject_id),
                                          device_data.header['patientcode'],
                                          subject_id],
                           'coll_id': [(device_data.header['patient_additional'] == coll_id),
                                       device_data.header['patient_additional'],
                                       coll_id],
                           'device_type': [(device_data.header['equipment'].split('_')[0] == device_type),
                                           device_data.header['equipment'].split('_')[0],
                                           device_type],
                           'device_id': ([(device_data.header['equipment'].split('_')[1] == device_id),
                                         device_data.header['equipment'].split('_')[1],
                                         device_id]
                                         if len(device_data.header['equipment'].split('_')) > 1
                                         else [False, '', device_id]),

                           'device_location': [(device_data.header['recording_additional'] == device_location),
                                               device_data.header['recording_additional'],
                                               device_location]}

            # generate message if any mismatches
            for key, value in header_comp.items():
                if not value[0]:
                    message(f"{subject_id}_{coll_id}_{device_type}_{device_location}:  {key} mismatch: " +
                            f"{value[1]} (header) != {value[2]} (device list)",
                            level='warning', display=(not quiet), log=log)

            if overwrite_header:

                message("Overwriting header from device list", level='info', display=(not quiet), log=log)

                device_data.header['patientcode'] = subject_id
                device_data.header['patient_additional'] = coll_id
                device_data.header['equipment'] = '_'.join([device_type, device_id])
                device_data.header['recording_additional'] = device_location

            if save:

                # TODO: option to rename files

                # create all file path variables
                device_file_base = os.path.splitext(device_file_name)[0]
                device_edf_name = '.'.join([device_file_base, 'edf'])
                standard_device_path = os.path.join(self.dirs['standard_device_edf'], device_type, device_edf_name)

                # check that all folders exist for data output files
                Path(os.path.dirname(standard_device_path)).mkdir(parents=True, exist_ok=True)

                message(f"Saving {standard_device_path}", level='info', display=(not quiet), log=log)

                # write device data as edf
                device_data.export_edf(file_path=standard_device_path)

            message("", level='info', display=(not quiet), log=log)

            self.devices.append(device_data)

        return True

    def nonwear(self, save=False, quiet=False, log=True):

        # process nonwear for all devices
        message("Processing nonwear...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        self.nonwear_times = pd.DataFrame()

        # crop final nonwear from all device data
        for index, row in tqdm(self.device_list.iterrows(), total=self.device_list.shape[0], leave=False,
                               desc='Processing nonwear'):

            if self.devices[index] is None:
                # TODO: ADD MESSAGE ?? ALSO IN OTHER MODULES
                continue

            # get info from device list
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_file_name = row['file_name']

            # TODO: search signal headers for signal labels
            accel_x_sig = 0
            accel_y_sig = 1
            accel_z_sig = 2
            temperature_sig = 3

            # TODO: call different algorithm based on device_type or signals available??
            # TODO: log algorithm used

            nonwear_times, nonwear_array = nwnonwear.vert_nonwear(
                                                        x_values=self.devices[index].signals[accel_x_sig],
                                                        y_values=self.devices[index].signals[accel_y_sig],
                                                        z_values=self.devices[index].signals[accel_z_sig],
                                                        temperature_values=self.devices[index].signals[temperature_sig],
                                                        quiet=quiet)

            bout_count = nonwear_times.shape[0]

            message(f"Detected {bout_count} nonwear bouts for {device_type} {device_location}",
                    level='info', display=(not quiet), log=log)

            # convert datapoints to times
            start_date = self.devices[index].header['startdate']
            sample_rate = self.devices[index].signal_headers[accel_x_sig]['sample_rate']

            start_times = []
            end_times = []

            for nw_index, nw_row in nonwear_times.iterrows():
                start_times.append(start_date + dt.timedelta(seconds=(nw_row['start_datapoint'] / sample_rate)))
                end_times.append(start_date + dt.timedelta(seconds=(nw_row['end_datapoint'] / sample_rate)))

            nonwear_times['start_time'] = start_times
            nonwear_times['end_time'] = end_times

            nonwear_times['subject_id'] = subject_id
            nonwear_times['coll_id'] = coll_id
            nonwear_times['device_type'] = device_type
            nonwear_times['device_location'] = device_location

            # reorder columns
            nonwear_times = nonwear_times[['subject_id', 'coll_id', 'device_type', 'device_location',
                                          'start_time', 'end_time']]

            # append to collection attribute
            self.nonwear_times = self.nonwear_times.append(nonwear_times, ignore_index=True)

            if save:

                # create all file path variables
                device_file_base = os.path.splitext(device_file_name)[0]
                nonwear_csv_name = '.'.join(['_'.join([device_file_base, "NONWEAR"]), "csv"])
                nonwear_csv_path = os.path.join(self.dirs['standard_nonwear_times'], device_type, nonwear_csv_name)

                Path(os.path.dirname(nonwear_csv_path)).mkdir(parents=True, exist_ok=True)

                message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log)

                nonwear_times.to_csv(nonwear_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log)

        return True

    def crop(self, save=False, quiet=False, min_duration=1, max_time_to_eof=20, log=True):

        message("Cropping final nonwear...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

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
            else:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No nonwear data",
                        level='warning', display=(not quiet), log=log)

            # only crop if last nonwear ends within 20 minutes of end of file
            is_cropped = ((nonwear_duration >= dt.timedelta(minutes=min_duration)) &
                          (nonwear_time_to_eof <= dt.timedelta(minutes=max_time_to_eof)))

            # set new file end time to which to crop
            new_start_time = start_time
            new_end_time = last_nonwear['start_time'].item() if is_cropped else end_time

            crop_duration = end_time - new_end_time

            message(f"Cropping {crop_duration} from {device_type} {device_location}",
                    level='info', display=(not quiet), log=log)

            self.devices[index].crop(new_start_time, new_end_time)

            if save:

                # create all file path variables
                device_file_base = os.path.splitext(device_file_name)[0]
                device_edf_name = '.'.join([device_file_base, 'edf'])

                cropped_device_path = os.path.join(self.dirs['cropped_device_edf'], device_type, device_edf_name)

                # check that all folders exist for data output files
                Path(os.path.dirname(cropped_device_path)).mkdir(parents=True, exist_ok=True)

                message(f"Saving {cropped_device_path}", level='info', display=(not quiet), log=log)

                # write cropped device data as edf
                self.devices[index].export_edf(file_path=cropped_device_path)

            message("", level='info', display=(not quiet), log=log)

        return True

    def save_sensors(self, quiet=False, log=True):

        message("Separating sensors from devices...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

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

                message(f"Saving {sen_path}", level='info', display=(not quiet), log=log)

                self.devices[index].export_edf(file_path=sen_path, sig_nums_out=sen_channels)

            message("", level='info', display=(not quiet), log=log)

        return True


def message(msg, level='info', display=True, log=True):

    level_switch = {'debug': lambda: logging.debug(msg),
                    'info': lambda: logging.info(msg),
                    'warning': lambda: logging.warning(msg),
                    'error': lambda: logging.error(msg),
                    'critical': lambda: logging.critical(msg)}

    if display:
        print(msg + "\n")

    if log:
        func = level_switch.get(level, lambda: 'Invalid')
        func()
