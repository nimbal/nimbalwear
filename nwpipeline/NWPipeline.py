import os
import datetime as dt
from pathlib import Path
import logging
import traceback

from tqdm import tqdm
import pandas as pd
import nwdata
import nwnonwear
from nwpipeline import __version__
import nwgait
from functools import wraps


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
        self.log_file_path = os.path.join(self.dirs['logs'], "processing.log")

        # TODO: check for required files (raw data, device_list)

        # read device list
        self.device_list = pd.read_csv(self.device_list_path, dtype=str).fillna('')

        # initialize folder structure
        for key, value in self.dirs.items():
            Path(value).mkdir(parents=True, exist_ok=True)

    def run(self, subject_ids=None, coll_ids=None, single_stage=None, overwrite_header=False, quiet=False, log=True):

        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=self.log_file_path,
                            level=logging.INFO)

        message("\n\n", level='info', display=(not quiet), log=log)
        message(f"---- Start processing pipeline ----------------------------------------------",
                level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        # if no subject_ids passed then do all
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        # if no coll_ids passed then do all
        if coll_ids is None:
            coll_ids = self.get_coll_ids()

        message(f"Version: {__version__}", level='info', display=(not quiet), log=log)
        message(f"Subjects: {subject_ids}", level='info', display=(not quiet), log=log)
        message(f"Collections: {coll_ids}", level='info', display=(not quiet), log=log)
        if single_stage is not None:
            message(f"Single stage: {single_stage}", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        for subject_id in tqdm(subject_ids, desc="Processing subjects", leave=True):

            for coll_id in tqdm(coll_ids, desc="Processing collections", leave=False):

                message(f"---- Subject {subject_id}, Collection {coll_id} --------", level='info', display=(not quiet),
                        log=log)
                message("", level='info', display=(not quiet), log=log)
                
                try:
                    # get devices for this collection from device_list
                    coll_device_list_df = self.device_list.loc[(self.device_list['subject_id'] == subject_id) &
                                                               (self.device_list['coll_id'] == coll_id)]
                    coll_device_list_df.reset_index(inplace=True, drop=True)
    
                    # construct collection class and process
                    coll = NWCollection(subject_id=subject_id, coll_id=coll_id, device_list=coll_device_list_df,
                                        dirs=self.dirs)
                    coll.process(single_stage=single_stage, overwrite_header=overwrite_header, min_crop_duration=3,
                                 max_crop_time_to_eof=20, quiet=quiet, log=log)
                except:
                    tb = traceback.format_exc()
                    message(tb, level='error', display=(not quiet), log=log)

        message("---- End ----------------------------------------------\n", level='info', display=(not quiet), log=log)

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

    device_locations = {'left_ankle': ['LA', 'LEFTANKLE', 'LANKLE'],
                        'right_ankle': ['RA', 'RIGHTANKLE', 'RANKLE']}

    devices = []
    nonwear_times = pd.DataFrame()
    bout_times = pd.DataFrame()
    step_times = pd.DataFrame()

    def __init__(self, subject_id, coll_id, device_list, dirs):

        self.subject_id = subject_id
        self.coll_id = coll_id
        self.device_list = device_list
        self.dirs = dirs
        self.status_path = os.path.join(self.dirs['meta'], 'status.csv')
        # the keys are the same as the function names
        self.coll_status = {
            'nwcollection_id': f'{self.subject_id}_{self.coll_id}',
            'read': '',
            'nonwear': '',
            'read_nonwear': '',
            'crop': '',
            'save_sensors': '',
            'gait': ''
        }
        self.status_df = pd.read_csv(self.status_path) if os.path.exists(self.status_path) else pd.DataFrame(columns=self.coll_status.keys())
    
    def coll_status(f):
        @wraps(f)
        def coll_status_wrapper(self, *args, **kwargs):
            if self.coll_status['nwcollection_id'] in self.status_df['nwcollection_id'].values:
                index = self.status_df.loc[self.status_df['nwcollection_id'] == self.coll_status['nwcollection_id']].index[0]
            else:
                index = (self.status_df.index.max() + 1)
            
            try:
                res = f(self, *args, **kwargs)
                self.coll_status[f.__name__] = 'Success'
                return res
            except Exception as e:
                self.coll_status[f.__name__] = f'Failed: {str(e)}'
                raise e
            finally:
                self.status_df.loc[index, list(self.coll_status.keys())] = list(self.coll_status.values())
                self.status_df.to_csv(self.status_path, index=False)
        return coll_status_wrapper

    def process(self, single_stage=None, overwrite_header=False, min_crop_duration=1, max_crop_time_to_eof=20,
                quiet=False, log=True):
        """Processes the collection

        Args:
            single_stage (str): None, 'read', 'nonwear', 'crop', 'save_sensors'
            ...
        Returns:
            True if successful, False otherwise.
        """

        # read data from all devices in collection
        self.read(single_stage=single_stage, overwrite_header=overwrite_header, save=True, quiet=quiet, log=log)

        # data integrity ??

        # synchronize devices

        # process nonwear for all devices
        if single_stage in [None, 'nonwear']:
            self.nonwear(save=True, quiet=quiet, log=log)

        if single_stage == 'crop':
            self.read_nonwear(quiet=quiet, log=log)

        # crop final nonwear
        if single_stage in [None, 'crop']:
            self.crop(save=True, min_duration=min_crop_duration, max_time_to_eof=max_crop_time_to_eof, quiet=quiet,
                      log=log)

        # save sensor edf files
        if single_stage in [None, 'save_sensors']:
            self.save_sensors(quiet=quiet, log=log)

        # process posture

        # process activity levels

        # process gait
        if single_stage in [None, 'gait']:
            self.gait(save=True, quiet=quiet, log=log)

        # process sleep

        return True

    @coll_status
    def read(self, single_stage=None, overwrite_header=False, save=False, rename_file=False, quiet=False, log=True):

        # TODO: for single stage, only read needed devices?

        message("Reading device data from files...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        import_switch = {'EDF': lambda: device_data.import_edf(device_file_path, quiet=quiet),
                         'GNAC': lambda: device_data.import_gnac(device_file_path, correct_drift=True, quiet=quiet),
                         'BITF': lambda: device_data.import_bitf(device_file_path),
                         'NONW': lambda: device_data.import_nonw(device_file_path, quiet=quiet)}

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
            device_file_base = os.path.splitext(device_file_name)[0]
            device_edf_name = '.'.join([device_file_base, 'edf'])

            if single_stage in [None, 'read']:

                device_file_path = os.path.join(self.dirs['raw'], device_type, device_file_name)
                import_func = import_switch.get(device_type, lambda: 'Invalid')

            elif single_stage in ['nonwear', 'crop']:

                device_file_path = os.path.join(self.dirs['standard_device_edf'], device_type, device_edf_name)
                import_func = import_switch.get('EDF', lambda: 'Invalid')

            else:

                device_file_path = os.path.join(self.dirs['cropped_device_edf'], device_type, device_edf_name)
                import_func = import_switch.get('EDF', lambda: 'Invalid')

            # check that data file exists
            if not os.path.isfile(device_file_path):
                message(
                    f"{subject_id}_{coll_id}_{device_type}_{device_location}: {device_file_path} does not exist",
                    level='warning', display=(not quiet), log=log)
                self.devices.append(None)
                continue

            # import data to device data object
            message(f"Reading {device_file_path}", level='info', display=(not quiet), log=log)

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

            if single_stage in [None, 'read'] and save:

                # TODO: option to rename files

                # create all file path variables
                standard_device_path = os.path.join(self.dirs['standard_device_edf'], device_type, device_edf_name)

                # check that all folders exist for data output files
                Path(os.path.dirname(standard_device_path)).mkdir(parents=True, exist_ok=True)

                message(f"Saving {standard_device_path}", level='info', display=(not quiet), log=log)

                # write device data as edf
                device_data.export_edf(file_path=standard_device_path)

            message("", level='info', display=(not quiet), log=log)

            self.devices.append(device_data)

        return True

    @coll_status 
    def nonwear(self, save=False, quiet=False, log=True):

        # process nonwear for all devices
        message("Detecting non-wear...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        self.nonwear_times = pd.DataFrame()

        # detect nonwear for each device
        for index, row in tqdm(self.device_list.iterrows(), total=self.device_list.shape[0], leave=False,
                               desc='Detecting non-wear'):

            # get info from device list
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_file_name = row['file_name']

            # TODO: Add nonwear detection for other devices

            if not device_type == 'GNAC':
                message(f"Cannot detect non-wear for {device_type}_{device_location}",
                        level='info', display=(not quiet), log=log)
                message("", level='info', display=(not quiet), log=log)
                continue

            # check for data loaded
            if self.devices[index] is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log)
                message("", level='info', display=(not quiet), log=log)
                continue

            # TODO: search signal headers for signal labels
            accel_x_sig = self.devices[index].get_signal_index('Accelerometer x')
            accel_y_sig = self.devices[index].get_signal_index('Accelerometer y')
            accel_z_sig = self.devices[index].get_signal_index('Accelerometer z')
            temperature_sig = self.devices[index].get_signal_index('Temperature')

            # TODO: call different algorithm based on device_type or signals available??
            # TODO: log algorithm used

            nonwear_times, nonwear_array = nwnonwear.vert_nonwear(
                                                        x_values=self.devices[index].signals[accel_x_sig],
                                                        y_values=self.devices[index].signals[accel_y_sig],
                                                        z_values=self.devices[index].signals[accel_z_sig],
                                                        temperature_values=self.devices[index].signals[temperature_sig],
                                                        quiet=quiet)
            algorithm_name = 'Vert algorithm'

            bout_count = nonwear_times.shape[0]

            message(f"Detected {bout_count} nonwear bouts for {device_type} {device_location} ({algorithm_name})",
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

    @coll_status
    def read_nonwear(self, quiet=False, log=True):

        # read nonwear data for all devices
        message("Reading non-wear data from files...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        self.nonwear_times = pd.DataFrame()

        # detect nonwear for each device
        for index, row in tqdm(self.device_list.iterrows(), total=self.device_list.shape[0], leave=False,
                               desc='Reading all non-wear data'):

            # get info from device list
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_file_name = row['file_name']

            device_file_base = os.path.splitext(device_file_name)[0]
            nonwear_csv_name = '.'.join(['_'.join([device_file_base, "NONWEAR"]), "csv"])
            nonwear_csv_path = os.path.join(self.dirs['standard_nonwear_times'], device_type, nonwear_csv_name)

            if not os.path.isfile(nonwear_csv_path):
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: {nonwear_csv_path} does not exist",
                        level='warning', display=(not quiet), log=log)
                message("", level='info', display=(not quiet), log=log)
                self.devices.append(None)
                continue

            message(f"Reading {nonwear_csv_path}", level='info', display=(not quiet), log=log)

            # read nonwear csv file
            nonwear_times = pd.read_csv(nonwear_csv_path, dtype=str)
            nonwear_times['start_time'] = pd.to_datetime(nonwear_times['start_time'], format='%Y-%m-%d %H:%M:%S')
            nonwear_times['end_time'] = pd.to_datetime(nonwear_times['end_time'], format='%Y-%m-%d %H:%M:%S')

            # append to collection attribute
            self.nonwear_times = self.nonwear_times.append(nonwear_times, ignore_index=True)

            message("", level='info', display=(not quiet), log=log)

        return True

    @coll_status
    def crop(self, save=False, quiet=False, min_duration=1, max_time_to_eof=20, log=True):

        message("Cropping final nonwear...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        # crop final nonwear from all device data
        for index, row in tqdm(self.device_list.iterrows(), total=self.device_list.shape[0], leave=False,
                               desc='Cropping final nonwear'):

            # get info from device list
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_file_name = row['file_name']

            if self.devices[index] is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log)
                continue

            last_nonwear = pd.DataFrame()

            if not self.nonwear_times.empty:

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

    @coll_status
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

    @coll_status
    def gait(self, save=False, quiet=False, log=True):

        message("Detecting steps and walking bouts...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        # TODO: these device locations only work for ReMiNDD data
        l_file_index = self.device_list.loc[self.device_list['device_location'] == 'LA'].index.values
        r_file_index = self.device_list.loc[self.device_list['device_location'] == 'RA'].index.values

        if not (l_file_index or r_file_index):
            raise Exception(f'{self.subject_id}_{self.coll_id}: No left or right ankle file found')

        # set indices and handles case if ankle data is missing
        l_file_index = l_file_index[0] if l_file_index else r_file_index[0]
        r_file_index = r_file_index[0] if r_file_index else l_file_index[0]

        # find accelerometer indices
        assert self.device_list.loc[l_file_index, 'device_type'] == self.device_list.loc[r_file_index, 'device_type']
        device_type = self.device_list.loc[l_file_index, 'device_type']
        accel_index = self.sensors_switch[device_type].index('ACCELEROMETER')
        sig_indices = self.sensor_channels_switch[device_type][accel_index]

        # get ankle files and only take accelerometer signals
        l_file = self.devices[l_file_index]
        r_file = self.devices[r_file_index]
        l_file.signals = [l_file.signals[i] for i in sig_indices]
        r_file.signals = [r_file.signals[i] for i in sig_indices]

        # TODO: should do same above for signal_headers even if not used

        # checks to see if files exist
        if not (l_file and r_file):
            raise Exception(f'{self.subject_id}_{self.coll_id}: Could not find nwgait object from index')

        # run gait algorithm to find bouts
        wb = nwgait.WalkingBouts(l_file, r_file, left_kwargs={'axis': 1}, right_kwargs={'axis': 1})

        # save bout times
        self.bout_times = wb.export_bouts()
        self.step_times = wb.export_steps()

        # TODO: add info to log about steps and bouts detected

        if save:
            # create all file path variables
            bouts_csv_name = '.'.join(['_'.join([self.subject_id, self.coll_id, "GAIT_BOUTS"]), "csv"])
            steps_csv_name = '.'.join(['_'.join([self.subject_id, self.coll_id, "GAIT_STEPS"]), "csv"])
            bouts_csv_path = os.path.join(self.dirs['gait'], bouts_csv_name)
            steps_csv_path = os.path.join(self.dirs['gait'], steps_csv_name)

            message(f"Saving {bouts_csv_path}", level='info', display=(not quiet), log=log)
            self.bout_times.to_csv(bouts_csv_path, index=False)

            message(f"Saving {steps_csv_path}", level='info', display=(not quiet), log=log)
            self.step_times.to_csv(steps_csv_path, index=False)

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
