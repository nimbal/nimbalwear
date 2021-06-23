import os
import datetime as dt
from pathlib import Path
import logging
import traceback
from functools import wraps
import json

from tqdm import tqdm
import pandas as pd
import numpy as np
import nwdata
import nwnonwear
from nwpipeline import __version__
import nwgait
import nwactivity
import nwsleep


class NWPipeline:

    def __init__(self, study_dir):

        # initialize folder structure
        self.study_dir = os.path.abspath(study_dir)

        # get study code
        self.study_code = os.path.basename(self.study_dir)

        self.dirs = {'study': '',
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
                     'epoch_activity': 'analyzed/activity/epoch_activity',
                     'daily_activity': 'analyzed/activity/daily_activity',
                     'gait': 'analyzed/gait',
                     'gait_steps': 'analyzed/gait/steps',
                     'gait_bouts': 'analyzed/gait/bouts',
                     'daily_gait': 'analyzed/gait/daily_gait',
                     'sleep': 'analyzed/sleep',
                     'sptw': 'analyzed/sleep/sptw',
                     'sleep_bouts': 'analyzed/sleep/sleep_bouts',
                     'daily_sleep': 'analyzed/sleep/daily_sleep'}

        self.dirs = {key: os.path.join(self.study_dir, value) for key, value in self.dirs.items()}

        # pipeline data files
        self.device_info_path = os.path.join(self.dirs['meta'], 'devices.csv')
        self.subject_info_path = os.path.join(self.dirs['meta'], 'subjects.csv')
        self.log_file_path = os.path.join(self.dirs['logs'], "processing.log")

        with open(os.path.join(Path(__file__).parent.absolute(),'data_dicts.json')) as f:
            self.data_dicts = json.loads(f.read())

        # TODO: check for required files (raw data, device_list)

        # read device list
        self.device_info = pd.read_csv(self.device_info_path, dtype=str).fillna('')

        # read subject level info
        if os.path.exists(self.subject_info_path):
            self.subject_info = pd.read_csv(self.subject_info_path, dtype=str).fillna('')
        else:
            self.subject_info = None

        # TODO: Check devices.csv and subjects.csv integrity
        # - ensure study code same for all rows (required) and matches study_dir (warning)
        # - unique combo of study, subject, coll, device type, device location (blanks allowed if still unique)
        # - ensure no missing file names

        # initialize folder structure
        for key, value in self.dirs.items():
            Path(value).mkdir(parents=True, exist_ok=True)
            # add data dictionary
            if key in self.data_dicts:
                df = pd.DataFrame(self.data_dicts[key], index=["column_label", "description", "type", "values"]).T
                p = os.path.join(value, f'{key}_dict.csv')
                df.to_csv(p)

    def run(self, subject_ids=None, coll_ids=None, single_stage=None, overwrite_header=False, min_crop_duration=3,
            max_crop_time_to_eof=20, activity_dominant=False, sleep_dominant=False, quiet=False, log=True):

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
        message(f"Study: {self.study_code}", level='info', display=(not quiet), log=log)
        message(f"Subjects: {subject_ids}", level='info', display=(not quiet), log=log)
        message(f"Collections: {coll_ids}", level='info', display=(not quiet), log=log)
        if single_stage is not None:
            message(f"Single stage: {single_stage}", level='info', display=(not quiet), log=log)
        if not isinstance(self.subject_info, pd.DataFrame):
            message("Missing subjects info file in meta folder `subjects.csv`", level='warning', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        for subject_id in tqdm(subject_ids, desc="Processing subjects", leave=True):

            for coll_id in tqdm(coll_ids, desc="Processing collections", leave=False):

                message("", level='info', display=(not quiet), log=log)
                message(f"---- Subject {subject_id}, Collection {coll_id} --------", level='info', display=(not quiet),
                        log=log)
                message("", level='info', display=(not quiet), log=log)
                
                try:
                    # get devices for this collection from device_list
                    coll_device_list_df = self.device_info.loc[(self.device_info['study_code'] == self.study_code) &
                                                               (self.device_info['subject_id'] == subject_id) &
                                                               (self.device_info['coll_id'] == coll_id)]
                    coll_device_list_df.reset_index(inplace=True, drop=True)

                    coll_subject_dict = {}
                    if isinstance(self.subject_info, pd.DataFrame):
                        coll_subject_df = self.subject_info.loc[(self.subject_info['study_code'] == self.study_code) &
                                                                    (self.subject_info['subject_id'] == subject_id)]
                        coll_subject_dict = coll_subject_df.iloc[0].to_dict() if coll_subject_df.shape[0] > 0 else {}
    
                    # construct collection class and process
                    coll = NWCollection(study_code=self.study_code, subject_id=subject_id, coll_id=coll_id,
                                        device_info=coll_device_list_df, subject_info=coll_subject_dict, dirs=self.dirs)
                    coll.process(single_stage=single_stage, overwrite_header=overwrite_header,
                                 min_crop_duration=min_crop_duration, max_crop_time_to_eof=max_crop_time_to_eof,
                                 activity_dominant=activity_dominant, sleep_dominant=sleep_dominant, quiet=quiet,
                                 log=log)
                except:
                    tb = traceback.format_exc()
                    message(tb, level='error', display=(not quiet), log=log)

                del coll

        message("---- End ----------------------------------------------\n", level='info', display=(not quiet), log=log)

    def get_subject_ids(self):

        subject_ids = self.device_info['subject_id'].unique()
        subject_ids.sort()

        return subject_ids

    def get_coll_ids(self):

        coll_ids = self.device_info['coll_id'].unique()
        coll_ids.sort()

        return coll_ids


class NWCollection:

    # TODO: should gyroscope be included with accelerometer when separating signals?

    sensors_switch = {'GNAC': ['ACCELEROMETER', 'TEMPERATURE', 'LIGHT', 'BUTTON'],
                      'AXV6': ['GYROSCOPE', 'ACCELEROMETER', 'LIGHT', 'TEMPERATURE'],
                      'BITF': ['ACCELEROMETER', 'ECG'],
                      'NONW': ['PLSOX']}

    sensor_channels_switch = {'GNAC': [[0, 1, 2], [3], [4], [5]],
                              'AXV6': [[0, 1, 2], [3, 4, 5], [6], [7]],
                              'BITF': [[1, 2, 3], [0]],
                              'NONW': [[0, 1]]}

    device_locations = {'left_ankle': ['LA', 'LEFTANKLE', 'LANKLE'],
                        'left_wrist': ['LW', 'LEFTWRIST', 'LWRIST'],
                        'right_wrist': ['RW', 'RIGHTWRIST', 'RWRIST'],
                        'right_ankle': ['RA', 'RIGHTANKLE', 'RANKLE']}

    devices = []
    nonwear_times = pd.DataFrame()
    bout_times = pd.DataFrame()
    step_times = pd.DataFrame()
    daily_activity = pd.DataFrame()
    epoch_activity = pd.DataFrame()
    sptw = pd.DataFrame()
    sleep_bouts = pd.DataFrame()
    daily_sleep = pd.DataFrame()

    def __init__(self, study_code, subject_id, coll_id, device_info, subject_info, dirs):

        self.study_code = study_code
        self.subject_id = subject_id
        self.coll_id = coll_id
        self.device_info = device_info
        self.dirs = dirs

        self.subject_info = subject_info if subject_info else {'dominant_hand': 'right'}

        self.status_path = os.path.join(self.dirs['meta'], 'status.csv')

    def coll_status(f):
        @wraps(f)
        def coll_status_wrapper(self, *args, **kwargs):

            # the keys are the same as the function names
            coll_status = {
                'nwcollection_id': f'{self.subject_id}_{self.coll_id}',
                'read': '',
                'nonwear': '',
                'crop': '',
                'save_sensors': '',
                'activity': '',
                'gait': '',
                'sleep': ''
            }

            status_df = pd.read_csv(self.status_path) if os.path.exists(self.status_path) \
                else pd.DataFrame(columns=coll_status.keys())

            if coll_status['nwcollection_id'] in status_df['nwcollection_id'].values:
                index = status_df.loc[status_df['nwcollection_id'] == coll_status['nwcollection_id']].index[0]
                coll_status = status_df.to_dict(orient='records')[index]
            else:
                index = (status_df.index.max() + 1)

            try:
                res = f(self, *args, **kwargs)
                coll_status[f.__name__] = 'Success'
                return res
            except NWException as e:
                coll_status[f.__name__] = f'Failed'
                message(str(e), level='error', display=(not kwargs['quiet']), log=kwargs['log'])
                message('', level='info', display=(not kwargs['quiet']), log=kwargs['log'])
            except Exception as e:
                coll_status[f.__name__] = f'Failed'
                raise e
            finally:
                status_df.loc[index, list(coll_status.keys())] = list(coll_status.values())
                status_df.to_csv(self.status_path, index=False)
        return coll_status_wrapper

    def process(self, single_stage=None, overwrite_header=False, min_crop_duration=1, max_crop_time_to_eof=20,
                activity_dominant=False, sleep_dominant=False, quiet=False, log=True):
        """Processes the collection

        Args:
            single_stage (str): None, 'read', 'nonwear', 'crop', 'save_sensors', 'activity', 'gait', 'sleep, 'posture'
            ...
        Returns:
            True if successful, False otherwise.
        """

        if single_stage in ['activity', 'gait', 'sleep']:
            self.required_devices(single_stage=single_stage, activity_dominant=activity_dominant,
                                  sleep_dominant=sleep_dominant, quiet=quiet, log=log)

        # read data from all devices in collection
        self.read(single_stage=single_stage, overwrite_header=overwrite_header, save=True, quiet=quiet, log=log)

        # data integrity ??

        # synchronize devices

        # process nonwear for all devices
        if single_stage in [None, 'nonwear']:
            self.nonwear(save=True, quiet=quiet, log=log)

        if single_stage in ['crop', 'sleep']:
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
        if single_stage in [None, 'activity']:
            self.activity(dominant=activity_dominant, save=True, quiet=quiet, log=log)

        # process gait
        if single_stage in [None, 'gait']:
            self.gait(save=True, quiet=quiet, log=log)

        # process sleep
        if single_stage in [None, 'sleep']:
            self.sleep(dominant=sleep_dominant, save=True, quiet=quiet, log=log)

        return True

    def required_devices(self, single_stage, activity_dominant=False, sleep_dominant=False, quiet=False, log=True):
        ''' Select only required devices for single stage processing.

        :param single_stage:
        :param quiet:
        :param log:
        :return:

        '''

        device_index = []

        if single_stage == 'activity':
            activity_device_index, activity_dominant = self.select_activity_device(dominant=activity_dominant)
            device_index += activity_device_index
        elif single_stage == 'gait':
            r_gait_device_index, l_gait_device_index = self.select_gait_device()
            device_index += r_gait_device_index + l_gait_device_index
        elif single_stage == 'sleep':
            sleep_device_index, sleep_dominant = self.select_sleep_device(dominant=sleep_dominant)
            device_index += sleep_device_index

        device_index = list(set(device_index))

        self.device_info = self.device_info.iloc[device_index]
        self.device_info.reset_index(inplace=True, drop=True)

        return True

    @coll_status
    def read(self, single_stage=None, overwrite_header=False, save=False, quiet=False, log=True):
        message("Reading device data from files...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        import_switch = {'EDF': lambda: device_data.import_edf(device_file_path, quiet=quiet),
                         'GNAC': lambda: device_data.import_gnac(device_file_path, correct_drift=True, quiet=quiet),
                         'AXV6': lambda: device_data.import_axiv(device_file_path, resample=True, quiet=quiet),
                         'BITF': lambda: device_data.import_bitf(device_file_path),
                         'NONW': lambda: device_data.import_nonw(device_file_path, quiet=quiet)}

        self.devices = []

        # read in all data files for one subject
        for index, row in tqdm(self.device_info.iterrows(), total=self.device_info.shape[0], leave=False,
                               desc='Reading all device data'):

            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_id = row['device_id']
            device_location = row['device_location']
            device_file_name = row['file_name']
            device_edf_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location]),
                                        "edf"])

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

            mismatch = False

            # check header against device list info
            header_comp = {'study_code': [(device_data.header['admincode'] == study_code),
                                          device_data.header['admincode'],
                                          self.study_code],
                           'subject_id': [(device_data.header['patientcode'] == subject_id),
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
                    mismatch = True

            if mismatch and overwrite_header:

                message("Overwriting header from device list", level='info', display=(not quiet), log=log)

                device_data.header['admincode'] = study_code
                device_data.header['patientcode'] = subject_id
                device_data.header['patient_additional'] = coll_id
                device_data.header['equipment'] = '_'.join([device_type, device_id])
                device_data.header['recording_additional'] = device_location

            if single_stage in [None, 'read'] and save:

                # TODO: option to rename files (or maybe not even an option, just do it)

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
        for index, row in tqdm(self.device_info.iterrows(), total=self.device_info.shape[0], leave=False,
                               desc='Detecting non-wear'):

            # get info from device list
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            # TODO: Add nonwear detection for other devices

            if not device_type in ['AXV6', 'GNAC']:
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

            accel_x_sig = self.devices[index].get_signal_index('Accelerometer x')
            accel_y_sig = self.devices[index].get_signal_index('Accelerometer y')
            accel_z_sig = self.devices[index].get_signal_index('Accelerometer z')
            temperature_sig = self.devices[index].get_signal_index('Temperature')

            # TODO: call different algorithm based on device_type or signals available??
            # TODO: log algorithm used

            nonwear_times, nonwear_array = nwnonwear.vert_nonwear(x_values=self.devices[index].signals[accel_x_sig],
                                                                  y_values=self.devices[index].signals[accel_y_sig],
                                                                  z_values=self.devices[index].signals[accel_z_sig],
                                                                  temperature_values=self.devices[index].signals[temperature_sig],
                                                                  accel_freq=self.devices[index].signal_headers[accel_x_sig]['sample_rate'],
                                                                  temperature_freq=self.devices[index].signal_headers[temperature_sig]['sample_rate'],
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

            # add study_code
            nonwear_times['study_code'] = study_code
            nonwear_times['subject_id'] = subject_id
            nonwear_times['coll_id'] = coll_id
            nonwear_times['device_type'] = device_type
            nonwear_times['device_location'] = device_location

            # reorder columns
            nonwear_times = nonwear_times[['study_code', 'subject_id', 'coll_id', 'device_type', 'device_location',
                                          'nonwear_bout_id', 'start_time', 'end_time']]

            # append to collection attribute
            self.nonwear_times = self.nonwear_times.append(nonwear_times, ignore_index=True)

            if save:

                # create all file path variables
                nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                       "NONWEAR"]),
                                             "csv"])
                nonwear_csv_path = os.path.join(self.dirs['standard_nonwear_times'], device_type, nonwear_csv_name)

                Path(os.path.dirname(nonwear_csv_path)).mkdir(parents=True, exist_ok=True)

                message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log)

                nonwear_times.to_csv(nonwear_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log)

        return True

    def read_nonwear(self, quiet=False, log=True):

        # read nonwear data for all devices
        message("Reading non-wear data from files...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        self.nonwear_times = pd.DataFrame()

        # detect nonwear for each device
        for index, row in tqdm(self.device_info.iterrows(), total=self.device_info.shape[0], leave=False,
                               desc='Reading all non-wear data'):

            # get info from device list
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id,
                                                   device_type, device_location, "NONWEAR"]),
                                         "csv"])
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
        for index, row in tqdm(self.device_info.iterrows(), total=self.device_info.shape[0], leave=False,
                               desc='Cropping final nonwear'):

            # get info from device list
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            if self.devices[index] is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log)
                continue

            last_nonwear = pd.DataFrame()

            if not self.nonwear_times.empty:

                # get last device nonwear period
                last_nonwear = self.nonwear_times.loc[(self.nonwear_times['study_code'] == study_code) &
                                                      (self.nonwear_times['subject_id'] == subject_id) &
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
                device_edf_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location]),
                                            "edf"])

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

        for index, row in tqdm(self.device_info.iterrows(), total=self.device_info.shape[0], leave=False,
                               desc='Saving sensor edfs'):

            if self.devices[index] is None:
                continue

            # get info from device list
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            # TODO: check that all device types in list are valid before running

            # evaluate device type cases
            sensors = self.sensors_switch.get(device_type, lambda: 'Invalid')
            sensor_channels = self.sensor_channels_switch.get(device_type, lambda: 'Invalid')

            # create all file path variables
            device_file_base = device_edf_name = '_'.join([study_code, subject_id, coll_id, device_type, device_location])
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
    def activity(self, dominant=False, save=False, quiet=False, log=True):

        message("Calculating activity levels...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        self.epoch_activity = pd.DataFrame()

        epoch_length = 15

        activity_device_index, dominant = self.select_activity_device(dominant=dominant)

        if len(activity_device_index) == 0:
            raise NWException(f"{self.subject_id}_{self.coll_id}: Wrist device not found in device list")

        activity_device_index = activity_device_index[0]

        # checks to see if files exist
        if not self.devices[activity_device_index]:
            raise NWException(f'{self.subject_id}_{self.coll_id}: Wrist device data is missing')

        accel_x_sig = self.devices[activity_device_index].get_signal_index('Accelerometer x')
        accel_y_sig = self.devices[activity_device_index].get_signal_index('Accelerometer y')
        accel_z_sig = self.devices[activity_device_index].get_signal_index('Accelerometer z')

        message(f"Calculating {epoch_length}-second epoch activity...", level='info', display=(not quiet), log=log)

        # TODO: need to allow variable epoch_length and dominant?
        self.epoch_activity = \
            nwactivity.calc_wrist_powell(x=self.devices[activity_device_index].signals[accel_x_sig],
                                         y=self.devices[activity_device_index].signals[accel_y_sig],
                                         z=self.devices[activity_device_index].signals[accel_z_sig],
                                         sample_rate=self.devices[activity_device_index].signal_headers[accel_x_sig]['sample_rate'],
                                         epoch_length=epoch_length, dominant=dominant, quiet=quiet)

        self.epoch_activity = self.identify_df(self.epoch_activity)

        #total_activity = nwactivity.sum_total_activity(epoch_intensity=epoch_intensity, epoch_length=epoch_length, quiet=quiet)

        message("Summarizing daily activity volumes...", level='info', display=(not quiet), log=log)
        self.daily_activity = nwactivity.sum_daily_activity(epoch_intensity=self.epoch_activity['intensity'], epoch_length=epoch_length,
                                            start_datetime=self.devices[activity_device_index].header['startdate'], quiet=quiet)

        self.daily_activity = self.identify_df(self.daily_activity)

        # TODO: more detailed log info about what was done, epochs, days, intensities?
        # TODO: info about algortihm and settings, device used, dominant vs non-dominant, in log, methods, or data table

        if save:

            # create all file path variables
            epoch_activity_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id,
                                                          self.coll_id, "EPOCH_ACTIVITY"]),
                                                "csv"])
            daily_activity_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id,
                                                          self.coll_id, "DAILY_ACTIVITY"]),
                                                "csv"])

            epoch_activity_csv_path = os.path.join(self.dirs['epoch_activity'], epoch_activity_csv_name)
            daily_activity_csv_path = os.path.join(self.dirs['daily_activity'], daily_activity_csv_name)

            Path(os.path.dirname(epoch_activity_csv_path)).mkdir(parents=True, exist_ok=True)
            Path(os.path.dirname(daily_activity_csv_path)).mkdir(parents=True, exist_ok=True)

            message(f"Saving {epoch_activity_csv_path}", level='info', display=(not quiet), log=log)
            self.epoch_activity.to_csv(epoch_activity_csv_path, index=False)

            message(f"Saving {daily_activity_csv_path}", level='info', display=(not quiet), log=log)
            self.daily_activity.to_csv(daily_activity_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log)

        return True

    @coll_status
    def gait(self, save=False, quiet=False, log=True, axis=1):
        message("Detecting steps and walking bouts...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        r_gait_device_index, l_gait_device_index = self.select_gait_device()

        if not (l_gait_device_index or r_gait_device_index):
            raise NWException(f'{self.subject_id}_{self.coll_id}: No left or right ankle device found in device list')

        # set indices and handles case if ankle data is missing
        l_gait_device_index = l_gait_device_index if l_gait_device_index else r_gait_device_index
        r_gait_device_index = r_gait_device_index if r_gait_device_index else l_gait_device_index

        l_gait_device_index = l_gait_device_index[0]
        r_gait_device_index = r_gait_device_index[0]

        # check to see that device_types match - comment because not necessary?
        # assert self.device_info.loc[l_gait_device_index, 'device_type'] == self.device_info.loc[r_gait_device_index, 'device_type']

        # checks to see if files exist
        if not (self.devices[l_gait_device_index] and self.devices[r_gait_device_index]):
            raise NWException(f'{self.subject_id}_{self.coll_id}: Either left or right ankle device data is missing')

        # convert inputs to objects as inputs
        l_accel_x_sig = self.devices[l_gait_device_index].get_signal_index('Accelerometer x')
        l_accel_y_sig = self.devices[l_gait_device_index].get_signal_index('Accelerometer y')
        l_accel_z_sig = self.devices[l_gait_device_index].get_signal_index('Accelerometer z')

        l_obj = nwgait.AccelReader.sig_init(raw_x=self.devices[l_gait_device_index].signals[l_accel_x_sig],
            raw_y=self.devices[l_gait_device_index].signals[l_accel_y_sig],
            raw_z=self.devices[l_gait_device_index].signals[l_accel_z_sig],
            startdate = self.devices[l_gait_device_index].header['startdate'],
            freq=self.devices[l_gait_device_index].signal_headers[l_accel_x_sig]['sample_rate'])

        r_accel_x_sig = self.devices[r_gait_device_index].get_signal_index('Accelerometer x')
        r_accel_y_sig = self.devices[r_gait_device_index].get_signal_index('Accelerometer y')
        r_accel_z_sig = self.devices[r_gait_device_index].get_signal_index('Accelerometer z')

        r_obj = nwgait.AccelReader.sig_init(raw_x=self.devices[r_gait_device_index].signals[r_accel_x_sig],
            raw_y=self.devices[r_gait_device_index].signals[r_accel_y_sig],
            raw_z=self.devices[r_gait_device_index].signals[r_accel_z_sig],
            startdate = self.devices[r_gait_device_index].header['startdate'],
            freq=self.devices[r_gait_device_index].signal_headers[r_accel_x_sig]['sample_rate'])

        # run gait algorithm to find bouts
        # TODO: Add progress bars instead of print statements??
        wb = nwgait.WalkingBouts(l_obj, r_obj, left_kwargs={'axis': axis}, right_kwargs={'axis': axis})

        # save bout times
        self.bout_times = wb.export_bouts()
        self.bout_times = self.identify_df(self.bout_times)
        self.step_times = wb.export_steps()
        self.step_times = self.identify_df(self.step_times)

        message(f"Detected {self.bout_times.shape[0]} gait bouts", level='info', display=(not quiet), log=log)
        message(f"Detected {self.step_times.shape[0]} steps", level='info', display=(not quiet), log=log)

        message("Summarizing daily gait analytics...", level='info', display=(not quiet), log=log)

        self.daily_gait = nwgait.WalkingBouts.daily_gait(self.bout_times)
        self.daily_gait = self.identify_df(self.daily_gait)

        # adjusting gait parameters
        bout_cols = ['study_code','subject_id','coll_id','gait_bout_num',
            'start_timestamp','end_timestamp', 'number_steps']
        self.bout_times = self.bout_times[bout_cols]
        step_cols = ['study_code','subject_id','coll_id','step_num',
            'gait_bout_num','foot','avg_speed','heel_strike_accel',
            'heel_strike_time','mid_swing_accel','mid_swing_time','step_length',
            'step_state','step_time','swing_start_accel','swing_start_time' ]
        self.step_times = self.step_times[step_cols]

        if save:
            # create all file path variables
            bouts_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id,
                                                 self.coll_id, "GAIT_BOUTS"]),
                                       "csv"])
            steps_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id,
                                                 self.coll_id, "GAIT_STEPS"]),
                                       "csv"])
            daily_gait_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id,
                                                 self.coll_id, "DAILY_GAIT"]),
                                       "csv"])
            bouts_csv_path = os.path.join(self.dirs['gait_bouts'], bouts_csv_name)
            steps_csv_path = os.path.join(self.dirs['gait_steps'], steps_csv_name)
            daily_gait_csv_path = os.path.join(self.dirs['daily_gait'], daily_gait_csv_name)

            message(f"Saving {bouts_csv_path}", level='info', display=(not quiet), log=log)
            self.bout_times.to_csv(bouts_csv_path, index=False)

            message(f"Saving {steps_csv_path}", level='info', display=(not quiet), log=log)
            self.step_times.to_csv(steps_csv_path, index=False)

            message(f"Saving {daily_gait_csv_path}", level='info', display=(not quiet), log=log)
            self.daily_gait.to_csv(daily_gait_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log)

        return True

    @coll_status
    def sleep(self, dominant=False, save=False, quiet=False, log=True):
        message("Detecting sleep...", level='info', display=(not quiet), log=log)
        message("", level='info', display=(not quiet), log=log)

        self.sptw = pd.DataFrame()
        self.sleep_bouts = pd.DataFrame()
        self.daily_sleep = pd.DataFrame()

        sleep_device_index, dominant = self.select_sleep_device(dominant=dominant)

        if len(sleep_device_index) == 0:
            raise NWException(f"{self.subject_id}_{self.coll_id}: Wrist device not found in device list")

        sleep_device_index = sleep_device_index[0]

        # checks to see if files exist
        if not self.devices[sleep_device_index]:
            raise NWException(f'{self.subject_id}_{self.coll_id}: Wrist device data is missing')

        accel_x_sig = self.devices[sleep_device_index].get_signal_index('Accelerometer x')
        accel_y_sig = self.devices[sleep_device_index].get_signal_index('Accelerometer y')
        accel_z_sig = self.devices[sleep_device_index].get_signal_index('Accelerometer z')

        # TODO: should sleep algorithm be modified if dominant vs non-dominant hand?
        # TODO: use Andrew Lim cutoffs for sleep detection?

        self.sptw, self.sleep_bouts = nwsleep.detect_sleep(x_values=self.devices[sleep_device_index].signals[accel_x_sig],
                                                           y_values=self.devices[sleep_device_index].signals[accel_y_sig],
                                                           z_values=self.devices[sleep_device_index].signals[accel_z_sig],
                                                           sample_rate=round(self.devices[sleep_device_index].signal_headers[accel_x_sig]['sample_rate']),
                                                           start_datetime=self.devices[sleep_device_index].header['startdate'],
                                                           z_abs_threshold=5, min_sleep_length=5)

        message(f"Detected {self.sptw.shape[0]} sleep period time windows", level='info', display=(not quiet), log=log)
        message(f"Detected {self.sleep_bouts.shape[0]} sleep bouts", level='info', display=(not quiet), log=log)

        message("Summarizing daily sleep analytics...", level='info', display=(not quiet), log=log)

        self.daily_sleep = nwsleep.sptw_stats(self.sptw, self.sleep_bouts, type='daily_long')

        self.sptw = self.identify_df(self.sptw)
        self.sleep_bouts = self.identify_df(self.sleep_bouts)
        self.daily_sleep = self.identify_df(self.daily_sleep)

        self.sptw.drop(['start_dp', 'end_dp'], axis='columns', inplace=True)
        self.sleep_bouts.drop(['start_dp', 'end_dp'], axis='columns', inplace=True)

        if save:

            # create all file path variables
            sptw_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id, self.coll_id, "SPTW"]), "csv"])
            sleep_bouts_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id, self.coll_id, "SLEEP_BOUTS"]),
                                             "csv"])

            daily_sleep_csv_name = '.'.join(['_'.join([self.study_code, self.subject_id, self.coll_id, "DAILY_SLEEP"]),
                                             "csv"])

            sptw_csv_path = os.path.join(self.dirs['sptw'], sptw_csv_name)
            sleep_bouts_csv_path = os.path.join(self.dirs['sleep_bouts'], sleep_bouts_csv_name)
            daily_sleep_csv_path = os.path.join(self.dirs['daily_sleep'], daily_sleep_csv_name)

            Path(os.path.dirname(sptw_csv_path)).mkdir(parents=True, exist_ok=True)
            Path(os.path.dirname(sleep_bouts_csv_path)).mkdir(parents=True, exist_ok=True)
            Path(os.path.dirname(daily_sleep_csv_path)).mkdir(parents=True, exist_ok=True)

            message(f"Saving {sptw_csv_path}", level='info', display=(not quiet), log=log)
            self.sptw.to_csv(sptw_csv_path, index=False)

            message(f"Saving {sleep_bouts_csv_path}", level='info', display=(not quiet), log=log)
            self.sleep_bouts.to_csv(sleep_bouts_csv_path, index=False)

            message(f"Saving {daily_sleep_csv_path}", level='info', display=(not quiet), log=log)
            self.daily_sleep.to_csv(daily_sleep_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log)

        return True

    def identify_df(self, df):
        df.insert(loc=0, column='study_code', value=self.study_code)
        df.insert(loc=1, column='subject_id', value=self.subject_id)
        df.insert(loc=2, column='coll_id', value=self.coll_id)
        return df

    def select_activity_device(self, dominant=False):

        # select which device to use for activity level

        # select eligible device types and locations
        activity_device_types = ['GNAC', 'AXV6']
        activity_locations = self.device_locations['right_wrist'] + self.device_locations['left_wrist']

        # get index of all eligible devices
        activity_device_index = self.device_info.loc[(self.device_info['device_type'].isin(activity_device_types)) &
                                                     (self.device_info['device_location'].isin(activity_locations))].index.values.tolist()

        # if multiple eligible devices we will try to choose one
        if len(activity_device_index) > 1:

            # if dominant hand is info is available we will choose based on dominant argument
            if self.subject_info['dominant_hand']:

                # select dominant or non-dominant based on argument
                if dominant:
                    wrist = 'right_wrist' if self.subject_info['dominant_hand'] == 'right' else 'left_wrist'
                else:
                    wrist = 'left_wrist' if self.subject_info['dominant_hand'] == 'right' else 'right_wrist'

                # select devices at locations based on dominance
                activity_locations = self.device_locations[wrist]
                activity_device_index = self.device_info.loc[
                    (self.device_info['device_type'].isin(activity_device_types)) &
                    (self.device_info['device_location'].isin(activity_locations))].index.values.tolist()

                # if still multiple eligible devices, take first one
                if len(activity_device_index) > 1:
                    activity_device_index = [activity_device_index[0]]

                # if no eligible devices, go back and take first one from list of all eligible
                elif len(activity_device_index) < 1:
                    activity_locations = self.device_locations['right_wrist'] + self.device_locations['left_wrist']
                    activity_device_index = self.device_info.loc[
                        (self.device_info['device_type'].isin(activity_device_types)) &
                        (self.device_info['device_location'].isin(activity_locations))].index.values.tolist()
                    activity_device_index = [activity_device_index[0]]

            # if no dominant hand info take first from list
            else:
                activity_device_index = [activity_device_index[0]]

        # if only one device determine, if it is dominant
        elif len(activity_device_index) == 1:

            # if dominant hand info is available we will determine dominance
            if self.subject_info['dominant_hand']:
                dominant_wrist = self.subject_info['dominant_hand'] + '_wrist'
                dominant = self.device_info.loc[activity_device_index]['device_location'].item() in \
                           self.device_locations[dominant_wrist]

            # if no dominant hand info available, assume dominant argument is correct

        return activity_device_index, dominant

    def select_gait_device(self):

        # select eligible device types and locations
        gait_device_types = ['GNAC', 'AXV6']
        r_gait_locations = self.device_locations['right_ankle']
        l_gait_locations = self.device_locations['left_ankle']

        # get index of all eligible devices
        r_gait_device_index = self.device_info.loc[(self.device_info['device_type'].isin(gait_device_types)) &
                                                     (self.device_info['device_location'].isin(r_gait_locations))].index.values.tolist()

        l_gait_device_index = self.device_info.loc[(self.device_info['device_type'].isin(gait_device_types)) &
                                                     (self.device_info['device_location'].isin(l_gait_locations))].index.values.tolist()

        #if more than one take the first
        if len(r_gait_device_index) > 1:
            r_gait_device_index = [r_gait_device_index[0]]
        if len(l_gait_device_index) > 1:
            l_gait_device_index = [l_gait_device_index[0]]

        return r_gait_device_index, l_gait_device_index

    def select_sleep_device(self, dominant=False):

        # select which device to use for activity level

        # select eligible device types and locations
        sleep_device_types = ['GNAC', 'AXV6']
        sleep_locations = self.device_locations['right_wrist'] + self.device_locations['left_wrist']

        # get index of all eligible devices
        sleep_device_index = self.device_info.loc[(self.device_info['device_type'].isin(sleep_device_types)) &
                                                     (self.device_info['device_location'].isin(
                                                         sleep_locations))].index.values.tolist()

        # if multiple eligible devices we will try to choose one
        if len(sleep_device_index) > 1:

            # if dominant hand is info is available we will choose based on dominant argument
            if self.subject_info['dominant_hand']:

                # select dominant or non-dominant based on argument
                if dominant:
                    wrist = 'right_wrist' if self.subject_info['dominant_hand'] == 'right' else 'left_wrist'
                else:
                    wrist = 'left_wrist' if self.subject_info['dominant_hand'] == 'right' else 'right_wrist'

                # select devices at locations based on dominance
                sleep_locations = self.device_locations[wrist]
                sleep_device_index = self.device_info.loc[(self.device_info['device_type'].isin(sleep_device_types)) &
                                                          (self.device_info['device_location'].isin(sleep_locations))].index.values.tolist()

                # if still multiple eligible devices, take first one
                if len(sleep_device_index) > 1:
                    sleep_device_index = [sleep_device_index[0]]

                # if no eligible devices, go back and take first one from list of all eligible
                elif len(sleep_device_index) < 1:
                    sleep_locations = self.device_locations['right_wrist'] + self.device_locations['left_wrist']
                    sleep_device_index = self.device_info.loc[(self.device_info['device_type'].isin(sleep_device_types)) &
                                                              (self.device_info['device_location'].isin(sleep_locations))].index.values.tolist()
                    sleep_device_index = [sleep_device_index[0]]

            # if no dominant hand info take first from list
            else:
                sleep_device_index = [sleep_device_index[0]]

        # if only one device determine, if it is dominant
        elif len(sleep_device_index) == 1:

            # if dominant hand info is available we will determine dominance
            if self.subject_info['dominant_hand']:
                dominant_wrist = self.subject_info['dominant_hand'] + '_wrist'
                dominant = self.device_info.loc[sleep_device_index]['device_location'].item() in self.device_locations[dominant_wrist]

            # if no dominant hand info available, assume dominant argument is correct

        return sleep_device_index, dominant


def message(msg, level='info', display=True, log=True):

    level_switch = {'debug': lambda: logging.debug(msg),
                    'info': lambda: logging.info(msg),
                    'warning': lambda: logging.warning(msg),
                    'error': lambda: logging.error(msg),
                    'critical': lambda: logging.critical(msg)}

    if display:
        print(msg)

    if log:
        func = level_switch.get(level, lambda: 'Invalid')
        func()


class NWException(Exception):
    """Hit NWException when an expected error occurs in pipeline"""
    pass
