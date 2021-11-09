import os
import shutil
import datetime as dt
from pathlib import Path
import logging
import traceback
from functools import wraps
import json

from tqdm import tqdm
import pandas as pd
import nwdata
import nwnonwear
from nwpipeline import __version__
import nwgait
import nwactivity
import nwsleep


class Pipeline:

    def __init__(self, study_dir, log_level=logging.INFO):

        self.quiet = False
        self.log = True

        # initialize folder structure
        self.study_dir = Path(study_dir)
        settings_path = self.study_dir / 'pipeline/settings.json'
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        if not settings_path.exists():
            settings_src = Path(__file__).parent.absolute() / 'settings.json'
            shutil.copy(settings_src, settings_path)

        # get study code
        self.study_code = self.study_dir.name

        # read json file
        with open(self.study_dir / 'pipeline/settings.json', 'r') as f:
            settings_json = json.load(f)

        self.dirs = settings_json['pipeline']['dirs']
        self.dirs = {key: self.study_dir / value for key, value in self.dirs.items()}

        # pipeline data files
        self.device_info_path = self.dirs['pipeline'] / 'devices.csv'
        self.subject_info_path = self.dirs['pipeline'] / 'subjects.csv'
        self.log_file_path = self.dirs['logs'] / 'processing.log'
        self.status_path = self.dirs['pipeline'] / 'status.csv'

        self.stages = settings_json['pipeline']['stages']
        self.sensors = settings_json['pipeline']['sensors']
        self.device_locations = settings_json['pipeline']['device_locations']
        self.module_settings = settings_json['modules']

        with open(Path(__file__).parent.absolute() / 'data_dicts.json', 'r') as f:
            self.data_dicts = json.load(f)

        # TODO: check for required files (raw data, device_list)

        # read device list
        self.device_info = pd.read_csv(self.device_info_path, dtype=str).fillna('')

        # read subject level info
        if self.subject_info_path.exists():
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
                df = pd.DataFrame(self.data_dicts[key])
                p = value / f'{key}_dict.csv'
                df.to_csv(p, index=False)

        fileh = logging.FileHandler(self.log_file_path, 'a')
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fileh.setFormatter(formatter)
        fileh.setLevel(log_level)

        logger = logging.getLogger(self.study_code)
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)
        logger.setLevel(log_level)
        logger.addHandler(fileh)

    def coll_status(f):
        @wraps(f)
        def coll_status_wrapper(self, *args, **kwargs):

            # the keys are the same as the function names
            coll_status = {
                'nwcollection_id': f"{kwargs['coll'].subject_id}_{kwargs['coll'].coll_id}",
                'convert': '',
                'nonwear': '',
                'crop': '',
                'save_sensors': '',
                'activity': '',
                'gait': '',
                'sleep': ''
            }

            status_df = pd.read_csv(self.status_path) if self.status_path.exists() \
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
                message(str(e), level='error', display=(not kwargs['quiet']), log=kwargs['log'],
                        logger_name=self.study_code)
                message('', level='info', display=(not kwargs['quiet']), log=kwargs['log'], logger_name=self.study_code)
            except Exception as e:
                coll_status[f.__name__] = f'Failed'
                raise e
            finally:
                status_df.loc[index, list(coll_status.keys())] = list(coll_status.values())
                status_df.to_csv(self.status_path, index=False)

        return coll_status_wrapper

    def run(self, collections=None, single_stage=None, quiet=False, log=True):
        """

        :param collections: list of tuples (subject_id, coll_id), default is None which will run all collections
        :param single_stage:

        :return:
        """

        self.quiet = quiet
        self.log = log

        message("\n\n", level='info', display=(not self.quiet), log=self.log, logger_name=self.study_code)
        message(f"---- Start processing pipeline ----------------------------------------------",
                level='info', display=(not self.quiet), log=self.log, logger_name=self.study_code)
        message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.study_code)

        # get all unique collections if none provided
        collections = self.get_collections() if collections is None else collections

        # TODO: ensure collections is a list of tuples

        message(f"Version: {__version__}", level='info', display=(not self.quiet), log=self.log,
                logger_name=self.study_code)
        message(f"Study: {self.study_code}", level='info', display=(not self.quiet), log=self.log,
                logger_name=self.study_code)
        message(f"Collections (Subject, Collection): {collections}", level='info', display=(not self.quiet),
                log=self.log, logger_name=self.study_code)

        if single_stage is not None:
            message(f"Single stage: {single_stage}", level='info', display=(not self.quiet), log=self.log,
                    logger_name=self.study_code)
        if not isinstance(self.subject_info, pd.DataFrame):
            message("Missing subjects info file in meta folder `subjects.csv`", level='warning',
                    display=(not self.quiet), log=self.log, logger_name=self.study_code)
        message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.study_code)

        for collection in tqdm(collections, desc="Processing collections", leave=True):

            subject_id = collection[0]
            coll_id = collection[1]

            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.study_code)
            message(f"---- Subject {subject_id}, Collection {coll_id} --------", level='info', display=(not self.quiet),
                    log=self.log, logger_name=self.study_code)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.study_code)

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
                    coll_subject_df.reset_index(inplace=True, drop=True)
                    coll_subject_dict = coll_subject_df.iloc[0].to_dict() if coll_subject_df.shape[0] > 0 else {}

                # construct collection class and process
                coll = Collection(study_code=self.study_code, subject_id=subject_id, coll_id=coll_id)

                coll.device_info = coll_device_list_df
                coll.subject_info = coll_subject_dict

                self.process_collection(coll=coll, single_stage=single_stage)

            except:
                tb = traceback.format_exc()
                message(tb, level='error', display=(not self.quiet), log=self.log, logger_name=self.study_code)

            del coll

        message("---- End ----------------------------------------------\n", level='info', display=(not self.quiet),
                log=self.log, logger_name=self.study_code)

    def process_collection(self, coll, single_stage=None):

        """Processes the collection

        Args:
            coll:
            single_stage (str): None, 'read', 'nonwear', 'crop', 'save_sensors', 'activity', 'gait', 'sleep, 'posture'
            ...
        Returns:
            True if successful, False otherwise.
        """

        if single_stage in ['activity', 'gait', 'sleep']:
            self.required_devices(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        # read data from all devices in collection
        self.read(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        # convert to edf
        if single_stage in [None, 'convert']:
            self.convert(coll=coll, quiet=self.quiet, log=self.log)

        # data integrity ??

        # synchronize devices

        # process nonwear for all devices
        if single_stage in [None, 'nonwear']:
            self.nonwear(coll=coll, quiet=self.quiet, log=self.log)

        if single_stage in ['crop', 'sleep']:
            self.read_nonwear(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        # crop final nonwear
        if single_stage in [None, 'crop']:
            self.crop(coll=coll, quiet=self.quiet, log=self.log)

        # save sensor edf files
        if single_stage in [None, 'save_sensors']:
            self.save_sensors(coll=coll, quiet=self.quiet, log=self.log)

        # process posture

        # process activity levels
        if single_stage in [None, 'activity']:
            self.activity(coll=coll, quiet=self.quiet, log=self.log)

        # process gait
        if single_stage in [None, 'gait']:
            self.gait(coll=coll, quiet=self.quiet, log=self.log, )

        # process sleep
        if single_stage in [None, 'sleep']:
            self.sleep(coll=coll, quiet=self.quiet, log=self.log)

        return True

    def required_devices(self, coll, single_stage, quiet=False, log=True):
        """ Select only required devices for single stage processing.

        :param coll:
        :param single_stage:
        :param quiet:
        :param log:
        :return:

        """

        device_index = []

        if single_stage == 'activity':
            activity_device_index, activity_dominant = self.select_activity_device(coll=coll)
            device_index += activity_device_index
        elif single_stage == 'gait':
            r_gait_device_index, l_gait_device_index = self.select_gait_device(coll=coll)
            device_index += r_gait_device_index + l_gait_device_index
        elif single_stage == 'sleep':
            sleep_device_index, sleep_dominant = self.select_sleep_device(coll=coll)
            device_index += sleep_device_index

        device_index = list(set(device_index))

        coll.device_info = coll.device_info.iloc[device_index]
        coll.device_info.reset_index(inplace=True, drop=True)

        return coll

    def read(self, coll, single_stage=None, quiet=False, log=True):

        message("Reading device data from files...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        overwrite_header = self.module_settings['read']['overwrite_header']

        # TODO: move to json or make autodetect?
        import_switch = {'EDF': lambda: device_data.import_edf(device_file_path, quiet=quiet),
                         'GNOR': lambda: device_data.import_geneactiv(device_file_path, correct_drift=True,
                                                                      quiet=quiet),
                         'AXV6': lambda: device_data.import_axivity(device_file_path, resample=True, quiet=quiet),
                         'BF18': lambda: device_data.import_bittium(device_file_path, quiet=quiet),
                         'BF36': lambda: device_data.import_bittium(device_file_path, quiet=quiet),
                         'NOWO': lambda: device_data.import_nonin(device_file_path, quiet=quiet)}

        coll.devices = []

        # read in all data files for one subject
        for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
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

            if single_stage in [None, 'convert']:

                device_file_path = self.dirs['device_raw'] / device_file_name
                import_func = import_switch.get(device_type, lambda: 'Invalid')

            elif single_stage in ['nonwear', 'crop']:

                device_file_path = self.dirs['device_edf_standard'] / device_edf_name
                import_func = import_switch.get('EDF', lambda: 'Invalid')

            else:

                device_file_path = self.dirs['device_edf_cropped'] / device_edf_name
                import_func = import_switch.get('EDF', lambda: 'Invalid')

            # check that data file exists
            if not device_file_path.exists():
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: {device_file_path} does not exist",
                        level='warning', display=(not quiet), log=log, logger_name=self.study_code)
                coll.devices.append(None)
                continue

            # import data to device data object
            message(f"Reading {device_file_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)

            device_data = nwdata.NWData()
            import_func()
            device_data.deidentify()

            mismatch = False

            # check header against device list info
            header_comp = {'study_code': [(device_data.header['admincode'] == study_code),
                                          device_data.header['admincode'],
                                          coll.study_code],
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
                            level='warning', display=(not quiet), log=log, logger_name=self.study_code)
                    mismatch = True

            if mismatch and overwrite_header:

                message("Overwriting header from device list", level='info', display=(not quiet), log=log,
                        logger_name=self.study_code)

                device_data.header['admincode'] = study_code
                device_data.header['patientcode'] = subject_id
                device_data.header['patient_additional'] = coll_id
                device_data.header['equipment'] = '_'.join([device_type, device_id])
                device_data.header['recording_additional'] = device_location

            message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

            coll.devices.append(device_data)

        return coll

    @coll_status
    def convert(self, coll, quiet=False, log=True):

        message("Converting device data to EDF...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        # read in all data files for one subject
        for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Converting device data to EDF'):

            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_edf_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location]),
                                        "edf"])

            # create all file path variables
            standard_device_path = self.dirs['device_edf_standard'] / device_edf_name

            # check that all folders exist for data output files
            standard_device_path.parent.mkdir(parents=True, exist_ok=True)

            message(f"Saving {standard_device_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)

            # write device data as edf
            coll.devices[index].export_edf(file_path=standard_device_path, quiet=quiet)

            message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    @coll_status
    def nonwear(self, coll, quiet=False, log=True):

        # process nonwear for all devices
        message("Detecting non-wear...", level='info', display=(not quiet), log=log, logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        save = self.module_settings['nonwear']['save']

        coll.nonwear_times = pd.DataFrame()

        # detect nonwear for each device
        for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Detecting non-wear'):

            # get info from device list
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            # TODO: Add nonwear detection for other devices

            if device_type not in ['AXV6', 'GNOR']:
                message(f"Cannot detect non-wear for {device_type}_{device_location}",
                        level='info', display=(not quiet), log=log, logger_name=self.study_code)
                message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)
                continue

            # check for data loaded
            if coll.devices[index] is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log, logger_name=self.study_code)
                message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)
                continue

            accel_x_sig = coll.devices[index].get_signal_index('Accelerometer x')
            accel_y_sig = coll.devices[index].get_signal_index('Accelerometer y')
            accel_z_sig = coll.devices[index].get_signal_index('Accelerometer z')
            temperature_sig = coll.devices[index].get_signal_index('Temperature')

            # TODO: call different algorithm based on device_type or signals available??
            # TODO: log algorithm used

            nonwear_times, nonwear_array = nwnonwear.vert_nonwear(x_values=coll.devices[index].signals[accel_x_sig],
                                                                  y_values=coll.devices[index].signals[accel_y_sig],
                                                                  z_values=coll.devices[index].signals[accel_z_sig],
                                                                  temperature_values=coll.devices[index].signals[temperature_sig],
                                                                  accel_freq=coll.devices[index].signal_headers[accel_x_sig]['sample_rate'],
                                                                  temperature_freq=coll.devices[index].signal_headers[temperature_sig]['sample_rate'],
                                                                  quiet=quiet)
            algorithm_name = 'Vert algorithm'

            bout_count = nonwear_times.shape[0]

            message(f"Detected {bout_count} nonwear bouts for {device_type} {device_location} ({algorithm_name})",
                    level='info', display=(not quiet), log=log, logger_name=self.study_code)

            # convert datapoints to times
            start_date = coll.devices[index].header['startdate']
            sample_rate = coll.devices[index].signal_headers[accel_x_sig]['sample_rate']

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
            coll.nonwear_times = coll.nonwear_times.append(nonwear_times, ignore_index=True)

            if save:

                # create all file path variables
                nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                       "NONWEAR"]),
                                             "csv"])
                nonwear_csv_path = self.dirs['nonwear_bouts_standard'] / nonwear_csv_name

                nonwear_csv_path.parent.mkdir(parents=True, exist_ok=True)

                message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.study_code)

                nonwear_times.to_csv(nonwear_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    def read_nonwear(self, coll, single_stage, quiet=False, log=True):

        # read nonwear data for all devices
        message("Reading non-wear data from files...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        if single_stage == 'crop':
            nonwear_csv_dir = self.dirs['nonwear_bouts_standard']
        else:
            nonwear_csv_dir = self.dirs['nonwear_bouts_cropped']

        coll.nonwear_times = pd.DataFrame()

        # detect nonwear for each device
        for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
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
            nonwear_csv_path = nonwear_csv_dir / nonwear_csv_name

            if not os.path.isfile(nonwear_csv_path):
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: {nonwear_csv_path} does not exist",
                        level='warning', display=(not quiet), log=log, logger_name=self.study_code)
                message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)
                #coll.devices.append(None)    THIS SHOULD NOT BE HERE? CUT AND PASTE ERROR?
                continue

            message(f"Reading {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)

            # read nonwear csv file
            nonwear_times = pd.read_csv(nonwear_csv_path, dtype=str)
            nonwear_times['start_time'] = pd.to_datetime(nonwear_times['start_time'], format='%Y-%m-%d %H:%M:%S')
            nonwear_times['end_time'] = pd.to_datetime(nonwear_times['end_time'], format='%Y-%m-%d %H:%M:%S')

            # append to collection attribute
            coll.nonwear_times = coll.nonwear_times.append(nonwear_times, ignore_index=True)

            message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    @coll_status
    def crop(self, coll, quiet=False, log=True):

        message("Detecting final device removal...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        min_duration = self.module_settings['crop']['min_duration']
        max_time_to_eof = self.module_settings['crop']['max_time_to_eof']
        save = self.module_settings['crop']['save']

        # crop final nonwear from all device data
        for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Detecting final device removal'):

            # get info from device list
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            if coll.devices[index] is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log, logger_name=self.study_code)
                continue

            # if there is nonwear data for any devices in this collection
            if not coll.nonwear_times.empty:

                # get nonwear indices for current device
                nonwear_idx = coll.nonwear_times.index[(coll.nonwear_times['study_code'] == study_code) &
                                                       (coll.nonwear_times['subject_id'] == subject_id) &
                                                       (coll.nonwear_times['coll_id'] == coll_id) &
                                                       (coll.nonwear_times['device_type'] == device_type) &
                                                       (coll.nonwear_times['device_location'] == device_location)]
                nonwear_idx = nonwear_idx.tolist()

                # if there is nonwear data for current device
                if len(nonwear_idx):

                    # get last nonwear period for current device
                    last_nonwear_idx = nonwear_idx[-1]
                    last_nonwear = coll.nonwear_times.loc[last_nonwear_idx]

                    # get time info from device data
                    start_time = coll.devices[index].header['startdate']
                    samples = len(coll.devices[index].signals[0])
                    sample_rate = coll.devices[index].signal_headers[0]['sample_rate']
                    duration = dt.timedelta(seconds=samples / sample_rate)
                    end_time = start_time + duration

                    # get duration and time to end of file of last nonwear
                    nonwear_duration = last_nonwear['end_time'] - last_nonwear['start_time']
                    nonwear_time_to_eof = end_time - last_nonwear['end_time']

                    # only crop if last nonwear ends within 20 minutes of end of file
                    early_removal = ((nonwear_duration >= dt.timedelta(minutes=min_duration)) &
                                    (nonwear_time_to_eof <= dt.timedelta(minutes=max_time_to_eof)))

                    #if removed early then crop
                    if early_removal:

                        # set new file end time to which to crop
                        new_start_time = start_time
                        new_end_time = last_nonwear['start_time']

                        crop_duration = end_time - new_end_time

                        message(f"Cropping {crop_duration} after final removal of {device_type} {device_location}",
                                level='info', display=(not quiet), log=log, logger_name=self.study_code)

                        coll.devices[index].crop(new_start_time, new_end_time, inplace=True)

                        # remove last non-wear from data frame
                        coll.nonwear_times.drop(index=last_nonwear_idx, inplace=True)

                    else:

                        message(f"No final removal of {device_type} {device_location} detected", level='info',
                                display=(not quiet), log=log, logger_name=self.study_code)

                else:
                    message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No nonwear data for device",
                            level='warning', display=(not quiet), log=log, logger_name=self.study_code)

            else:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No nonwear data for collection",
                        level='warning', display=(not quiet), log=log, logger_name=self.study_code)

            if save:

                # create all file path variables
                device_edf_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location]),
                                            "edf"])
                nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                       "NONWEAR"]),
                                             "csv"])

                cropped_device_path = self.dirs['device_edf_cropped'] / device_edf_name
                nonwear_csv_path = self.dirs['nonwear_bouts_cropped'] / nonwear_csv_name

                # check that all folders exist for data output files
                cropped_device_path.parent.mkdir(parents=True, exist_ok=True)
                nonwear_csv_path.parent.mkdir(parents=True, exist_ok=True)

                # write cropped device data as edf
                message(f"Saving {cropped_device_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.study_code)
                coll.devices[index].export_edf(file_path=cropped_device_path, quiet=quiet)

                # write nonwear times with cropped nonwear removed
                message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.study_code)
                coll.nonwear_times[coll.nonwear_times.index.isin(nonwear_idx)].to_csv(nonwear_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    @coll_status
    def save_sensors(self, coll, quiet=False, log=True):

        message("Separating sensors from devices...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Saving sensor edfs'):

            if coll.devices[index] is None:
                continue

            # get info from device list
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            device_file_base = '_'.join([study_code, subject_id, coll_id, device_type, device_location])

            # loop through supported sensor types
            for key in tqdm(self.sensors, leave=False, desc="Separating sensors"):

                # search for associated signals in current device
                sig_nums = []
                for sig_label in self.sensors[key]['signals']:
                    sig_num = coll.devices[index].get_signal_index(sig_label)

                    if sig_num is not None:
                        sig_nums.append(sig_num)

                # if signal labels from that sensor are present then save as sensor file
                if sig_nums:

                    sensor_edf_name = '.'.join(['_'.join([device_file_base, key.upper()]), 'edf'])
                    sensor_path = self.dirs['sensor_edf'] / sensor_edf_name
                    sensor_path.parent.mkdir(parents=True, exist_ok=True)

                    message(f"Saving {sensor_path}", level='info', display=(not quiet), log=log,
                            logger_name=self.study_code)

                    coll.devices[index].export_edf(file_path=sensor_path, sig_nums_out=sig_nums, quiet=quiet)

            message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    @coll_status
    def activity(self, coll, quiet=False, log=True):

        message("Calculating activity levels...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        save = self.module_settings['activity']['save']

        coll.epoch_activity = pd.DataFrame()

        epoch_length = 15

        activity_device_index, dominant = self.select_activity_device(coll=coll)

        if len(activity_device_index) == 0:
            raise NWException(f"{coll.subject_id}_{coll.coll_id}: Wrist device not found in device list")

        activity_device_index = activity_device_index[0]

        # checks to see if files exist
        if not coll.devices[activity_device_index]:
            raise NWException(f'{coll.subject_id}_{coll.coll_id}: Wrist device data is missing')

        accel_x_sig = coll.devices[activity_device_index].get_signal_index('Accelerometer x')
        accel_y_sig = coll.devices[activity_device_index].get_signal_index('Accelerometer y')
        accel_z_sig = coll.devices[activity_device_index].get_signal_index('Accelerometer z')

        message(f"Calculating {epoch_length}-second epoch activity...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)

        # TODO: need to allow variable epoch_length and dominant?
        coll.epoch_activity = \
            nwactivity.calc_wrist_powell(x=coll.devices[activity_device_index].signals[accel_x_sig],
                                         y=coll.devices[activity_device_index].signals[accel_y_sig],
                                         z=coll.devices[activity_device_index].signals[accel_z_sig],
                                         sample_rate=coll.devices[activity_device_index].signal_headers[accel_x_sig]['sample_rate'],
                                         epoch_length=epoch_length, dominant=dominant, quiet=quiet)

        coll.epoch_activity = self.identify_df(coll, coll.epoch_activity)

        #total_activity = nwactivity.sum_total_activity(epoch_intensity=epoch_intensity, epoch_length=epoch_length, quiet=quiet)

        message("Summarizing daily activity volumes...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        coll.daily_activity = \
            nwactivity.sum_daily_activity(epoch_intensity=coll.epoch_activity['intensity'],
                                          epoch_length=epoch_length,
                                          start_datetime=coll.devices[activity_device_index].header['startdate'],
                                          quiet=quiet)

        coll.daily_activity = self.identify_df(coll, coll.daily_activity)

        # TODO: more detailed log info about what was done, epochs, days, intensities?
        # TODO: info about algortihm and settings, device used, dominant vs non-dominant, in log, methods, or data table

        if save:

            # create all file path variables
            epoch_activity_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id,
                                                          coll.coll_id, "EPOCH_ACTIVITY"]),
                                                "csv"])
            daily_activity_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id,
                                                          coll.coll_id, "DAILY_ACTIVITY"]),
                                                "csv"])

            epoch_activity_csv_path = self.dirs['activity_epoch'] / epoch_activity_csv_name
            daily_activity_csv_path = self.dirs['activity_daily'] / daily_activity_csv_name

            epoch_activity_csv_path.parent.mkdir(parents=True, exist_ok=True)
            daily_activity_csv_path.parent.mkdir(parents=True, exist_ok=True)

            message(f"Saving {epoch_activity_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)
            coll.epoch_activity.to_csv(epoch_activity_csv_path, index=False)

            message(f"Saving {daily_activity_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)
            coll.daily_activity.to_csv(daily_activity_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    @coll_status
    def gait(self, coll, quiet=False, log=True):

        # TODO: axis needs to be set based on orientation of device

        message("Detecting steps and walking bouts...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        axis = self.module_settings['gait']['axis']
        save = self.module_settings['gait']['save']

        r_gait_device_index, l_gait_device_index = self.select_gait_device(coll=coll)

        if not (l_gait_device_index or r_gait_device_index):
            raise NWException(f'{coll.subject_id}_{coll.coll_id}: No left or right ankle device found in device list')

        # set indices and handles case if ankle data is missing
        l_gait_device_index = l_gait_device_index if l_gait_device_index else r_gait_device_index
        r_gait_device_index = r_gait_device_index if r_gait_device_index else l_gait_device_index

        l_gait_device_index = l_gait_device_index[0]
        r_gait_device_index = r_gait_device_index[0]

        # check to see that device_types match - comment because not necessary?
        # assert self.device_info.loc[l_gait_device_index, 'device_type'] == self.device_info.loc[r_gait_device_index, 'device_type']

        # checks to see if files exist
        if not (coll.devices[l_gait_device_index] and coll.devices[r_gait_device_index]):
            raise NWException(f'{coll.subject_id}_{coll.coll_id}: Either left or right ankle device data is missing')

        # convert inputs to objects as inputs
        l_accel_x_sig = coll.devices[l_gait_device_index].get_signal_index('Accelerometer x')
        l_accel_y_sig = coll.devices[l_gait_device_index].get_signal_index('Accelerometer y')
        l_accel_z_sig = coll.devices[l_gait_device_index].get_signal_index('Accelerometer z')

        l_obj = nwgait.AccelReader.sig_init(raw_x=coll.devices[l_gait_device_index].signals[l_accel_x_sig],
            raw_y=coll.devices[l_gait_device_index].signals[l_accel_y_sig],
            raw_z=coll.devices[l_gait_device_index].signals[l_accel_z_sig],
            startdate = coll.devices[l_gait_device_index].header['startdate'],
            freq=coll.devices[l_gait_device_index].signal_headers[l_accel_x_sig]['sample_rate'])

        r_accel_x_sig = coll.devices[r_gait_device_index].get_signal_index('Accelerometer x')
        r_accel_y_sig = coll.devices[r_gait_device_index].get_signal_index('Accelerometer y')
        r_accel_z_sig = coll.devices[r_gait_device_index].get_signal_index('Accelerometer z')

        r_obj = nwgait.AccelReader.sig_init(raw_x=coll.devices[r_gait_device_index].signals[r_accel_x_sig],
            raw_y=coll.devices[r_gait_device_index].signals[r_accel_y_sig],
            raw_z=coll.devices[r_gait_device_index].signals[r_accel_z_sig],
            startdate = coll.devices[r_gait_device_index].header['startdate'],
            freq=coll.devices[r_gait_device_index].signal_headers[r_accel_x_sig]['sample_rate'])

        # run gait algorithm to find bouts
        # TODO: Add progress bars instead of print statements??
        wb = nwgait.WalkingBouts(l_obj, r_obj, left_kwargs={'axis': axis}, right_kwargs={'axis': axis})

        # save bout times
        coll.bout_times = wb.export_bouts()
        coll.bout_times = self.identify_df(coll, coll.bout_times)

        # save step times
        coll.step_times = wb.export_steps()

        # compensate for export_steps returning blank DataFrame if no steps
        # TODO: Fix in nwgait to return columns
        if coll.step_times.empty:
            coll.step_times = pd.DataFrame(columns=['step_num', 'gait_bout_num', 'foot', 'avg_speed',
                                                    'heel_strike_accel', 'heel_strike_time', 'mid_swing_accel',
                                                    'mid_swing_time', 'step_length', 'step_state', 'step_time',
                                                    'swing_start_accel', 'swing_start_time'])

        coll.step_times = self.identify_df(coll, coll.step_times)


        message(f"Detected {coll.bout_times.shape[0]} gait bouts", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)
        message(f"Detected {coll.step_times[coll.step_times['step_state'] == 'success'].shape[0]} steps",
                level='info', display=(not quiet), log=log, logger_name=self.study_code)

        message("Summarizing daily gait analytics...", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)

        coll.daily_gait = nwgait.WalkingBouts.daily_gait(coll.bout_times)
        coll.daily_gait = self.identify_df(coll, coll.daily_gait)

        # adjusting gait parameters
        bout_cols = ['study_code', 'subject_id', 'coll_id', 'gait_bout_num', 'start_timestamp', 'end_timestamp',
                     'number_steps']
        coll.bout_times = coll.bout_times[bout_cols]

        step_cols = ['study_code','subject_id','coll_id','step_num', 'gait_bout_num', 'foot', 'avg_speed',
                     'heel_strike_accel', 'heel_strike_time', 'mid_swing_accel', 'mid_swing_time', 'step_length',
                     'step_state', 'step_time', 'swing_start_accel', 'swing_start_time']
        coll.step_times = coll.step_times[step_cols]

        if save:
            # create all file path variables
            bouts_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_BOUTS"]), "csv"])
            steps_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_STEPS"]), "csv"])
            steps_rej_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_STEPS_REJ"]),
                                           "csv"])
            daily_gait_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_DAILY"]),
                                            "csv"])

            bouts_csv_path = self.dirs['gait_bouts'] / bouts_csv_name
            steps_csv_path = self.dirs['gait_steps'] / steps_csv_name
            steps_rej_csv_path = self.dirs['gait_steps'] / steps_rej_csv_name
            daily_gait_csv_path = self.dirs['gait_daily'] / daily_gait_csv_name

            message(f"Saving {bouts_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.study_code)
            coll.bout_times.to_csv(bouts_csv_path, index=False)

            message(f"Saving {steps_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.study_code)
            coll.step_times[coll.step_times['step_state'] == 'success'].to_csv(steps_csv_path, index=False)

            message(f"Saving {steps_rej_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)
            coll.step_times[coll.step_times['step_state'] != 'success'].to_csv(steps_rej_csv_path, index=False)

            message(f"Saving {daily_gait_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)
            coll.daily_gait.to_csv(daily_gait_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    @coll_status
    def sleep(self, coll, quiet=False, log=True):

        message("Analyzing sleep...", level='info', display=(not quiet), log=log, logger_name=self.study_code)
        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        save = self.module_settings['sleep']['save']

        coll.sptw = pd.DataFrame()
        coll.sleep_bouts = pd.DataFrame()
        coll.daily_sleep = pd.DataFrame()

        sleep_device_index, dominant = self.select_sleep_device(coll=coll)

        if len(sleep_device_index) == 0:
            raise NWException(f"{coll.subject_id}_{coll.coll_id}: Wrist device not found in device list")

        sleep_device_index = sleep_device_index[0]

        # checks to see if files exist
        if not coll.devices[sleep_device_index]:
            raise NWException(f'{coll.subject_id}_{coll.coll_id}: Wrist device data is missing')

        accel_x_sig = coll.devices[sleep_device_index].get_signal_index('Accelerometer x')
        accel_y_sig = coll.devices[sleep_device_index].get_signal_index('Accelerometer y')
        accel_z_sig = coll.devices[sleep_device_index].get_signal_index('Accelerometer z')

        # get nonwear for sleep_device
        device_nonwear = coll.nonwear_times.loc[(coll.nonwear_times['study_code'] == coll.study_code) &
                                                (coll.nonwear_times['subject_id'] == coll.subject_id) &
                                                (coll.nonwear_times['coll_id'] == coll.coll_id) &
                                                (coll.nonwear_times['device_type'] == coll.device_info.iloc[sleep_device_index]['device_type']) &
                                                (coll.nonwear_times['device_location'] == coll.device_info.iloc[sleep_device_index]['device_location'])]

        # TODO: should sleep algorithm be modified if dominant vs non-dominant hand?

        coll.sptw, z_angle, z_angle_diff, z_sample_rate = nwsleep.detect_sptw(
            x_values=coll.devices[sleep_device_index].signals[accel_x_sig],
            y_values=coll.devices[sleep_device_index].signals[accel_y_sig],
            z_values=coll.devices[sleep_device_index].signals[accel_z_sig],
            sample_rate=round(coll.devices[sleep_device_index].signal_headers[accel_x_sig]['sample_rate']),
            start_datetime=coll.devices[sleep_device_index].header['startdate'],
            nonwear = device_nonwear)

        message(f"Detected {coll.sptw.shape[0]} sleep period time windows", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)

        sleep_t5a5 = nwsleep.detect_sleep_bouts(z_angle_diff=z_angle_diff, sptw=coll.sptw,
                                                      z_sample_rate=z_sample_rate,
                                                      start_datetime=coll.devices[sleep_device_index].header['startdate'],
                                                      z_abs_threshold=5, min_sleep_length=5)

        sleep_t5a5.insert(loc=2, column='bout_detect', value='t5a5')

        message(f"Detected {sleep_t5a5.shape[0]} sleep bouts (t5a5)", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)

        sleep_t8a4 = nwsleep.detect_sleep_bouts(z_angle_diff=z_angle_diff, sptw=coll.sptw,
                                                z_sample_rate=z_sample_rate,
                                                start_datetime=coll.devices[sleep_device_index].header['startdate'],
                                                z_abs_threshold=4, min_sleep_length=8)

        sleep_t8a4.insert(loc=2, column='bout_detect', value='t8a4')

        message(f"Detected {sleep_t8a4.shape[0]} sleep bouts (t8a4)", level='info', display=(not quiet), log=log,
                logger_name=self.study_code)

        coll.sleep_bouts = pd.concat([sleep_t5a5, sleep_t8a4])

        daily_sleep_t5a5 = nwsleep.sptw_stats(coll.sptw, sleep_t5a5, type='daily', sptw_inc=['long', 'all', 'sleep'])
        message(f"Summarized {daily_sleep_t5a5['sptw_inc'].value_counts()['long']} days of sleep analytics (t5a5)...",
                level='info', display=(not quiet), log=log, logger_name=self.study_code)

        daily_sleep_t8a4 = nwsleep.sptw_stats(coll.sptw, sleep_t8a4, type='daily', sptw_inc=['long', 'all', 'sleep'])
        message(f"Summarized {daily_sleep_t8a4['sptw_inc'].value_counts()['long']} days of sleep analytics (t8a4)...",
                level='info', display=(not quiet), log=log, logger_name=self.study_code)

        daily_sleep_t5a5.insert(loc=2, column='bout_detect', value='t5a5')
        daily_sleep_t8a4.insert(loc=2, column='bout_detect', value='t8a4')

        coll.daily_sleep = pd.concat([daily_sleep_t5a5, daily_sleep_t8a4])

        coll.sptw = self.identify_df(coll, coll.sptw)
        coll.sleep_bouts = self.identify_df(coll, coll.sleep_bouts)
        coll.daily_sleep = self.identify_df(coll, coll.daily_sleep)

        if save:

            # create all file path variables
            sptw_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "SPTW"]), "csv"])
            sleep_bouts_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "SLEEP_BOUTS"]),
                                             "csv"])

            daily_sleep_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "DAILY_SLEEP"]),
                                             "csv"])

            sptw_csv_path = self.dirs['sleep_sptw'] / sptw_csv_name
            sleep_bouts_csv_path = self.dirs['sleep_bouts'] / sleep_bouts_csv_name
            daily_sleep_csv_path = self.dirs['sleep_daily'] / daily_sleep_csv_name

            sptw_csv_path.parent.mkdir(parents=True, exist_ok=True)
            sleep_bouts_csv_path.parent.mkdir(parents=True, exist_ok=True)
            daily_sleep_csv_path.parent.mkdir(parents=True, exist_ok=True)

            message(f"Saving {sptw_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.study_code)
            coll.sptw.to_csv(sptw_csv_path, index=False)

            message(f"Saving {sleep_bouts_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)
            coll.sleep_bouts.to_csv(sleep_bouts_csv_path, index=False)

            message(f"Saving {daily_sleep_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.study_code)
            coll.daily_sleep.to_csv(daily_sleep_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.study_code)

        return coll

    def select_activity_device(self, coll):

        # select which device to use for activity level

        dominant = self.module_settings['activity']['dominant']

        device_info_copy = coll.device_info.copy()
        device_info_copy['device_location'] = [x.upper() for x in device_info_copy['device_location']]

        # select eligible device types and locations
        activity_device_types = ['GNOR', 'AXV6']
        activity_locations = self.device_locations['rwrist']['aliases'] + self.device_locations['lwrist']['aliases']

        # get index of all eligible devices
        activity_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(activity_device_types)) &
                                                     (device_info_copy['device_location'].isin(activity_locations))].index.values.tolist()

        # if multiple eligible devices we will try to choose one
        if len(activity_device_index) > 1:

            # if dominant hand is info is available we will choose based on dominant argument
            if coll.subject_info['dominant_hand'] in ['right', 'left']:

                # select dominant or non-dominant based on argument
                if dominant:
                    wrist = 'rwrist' if coll.subject_info['dominant_hand'] == 'right' else 'lwrist'
                else:
                    wrist = 'lwrist' if coll.subject_info['dominant_hand'] == 'right' else 'rwrist'

                # select devices at locations based on dominance
                activity_locations = self.device_locations[wrist]['aliases']
                activity_device_index = device_info_copy.loc[
                    (device_info_copy['device_type'].isin(activity_device_types)) &
                    (device_info_copy['device_location'].isin(activity_locations))].index.values.tolist()

                # if still multiple eligible devices, take first one
                if len(activity_device_index) > 1:
                    activity_device_index = [activity_device_index[0]]

                # if no eligible devices, go back and take first one from list of all eligible
                elif len(activity_device_index) < 1:
                    activity_locations = self.device_locations['rwrist']['aliases'] + self.device_locations['lwrist']['aliases']
                    activity_device_index = device_info_copy.loc[
                        (device_info_copy['device_type'].isin(activity_device_types)) &
                        (device_info_copy['device_location'].isin(activity_locations))].index.values.tolist()
                    activity_device_index = [activity_device_index[0]]

            # if no dominant hand info take first from list
            else:
                activity_device_index = [activity_device_index[0]]

        # if only one device determine, if it is dominant
        elif len(activity_device_index) == 1:

            # if dominant hand info is available we will determine dominance
            if coll.subject_info['dominant_hand'] in ['right', 'left']:
                dominant_wrist = coll.subject_info['dominant_hand'][0] + 'wrist'
                dominant = device_info_copy.loc[activity_device_index]['device_location'].item() in \
                           self.device_locations[dominant_wrist]['aliases']

            # if no dominant hand info available, assume dominant argument is correct

        return activity_device_index, dominant

    def select_gait_device(self, coll):

        device_info_copy = coll.device_info.copy()
        device_info_copy['device_location'] = [x.upper() for x in device_info_copy['device_location']]

        # select eligible device types and locations
        gait_device_types = ['GNOR', 'AXV6']
        r_gait_locations = self.device_locations['rankle']['aliases']
        l_gait_locations = self.device_locations['lankle']['aliases']

        # get index of all eligible devices
        r_gait_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(gait_device_types)) &
                                                     (device_info_copy['device_location'].isin(r_gait_locations))].index.values.tolist()

        l_gait_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(gait_device_types)) &
                                                     (device_info_copy['device_location'].isin(l_gait_locations))].index.values.tolist()

        #if more than one take the first
        if len(r_gait_device_index) > 1:
            r_gait_device_index = [r_gait_device_index[0]]
        if len(l_gait_device_index) > 1:
            l_gait_device_index = [l_gait_device_index[0]]

        return r_gait_device_index, l_gait_device_index

    def select_sleep_device(self, coll):

        # select which device to use for activity level

        dominant = self.module_settings['sleep']['dominant']

        device_info_copy = coll.device_info.copy()
        device_info_copy['device_location'] = [x.upper() for x in device_info_copy['device_location']]

        # select eligible device types and locations
        sleep_device_types = ['GNOR', 'AXV6']
        sleep_locations = self.device_locations['rwrist']['aliases'] + self.device_locations['lwrist']['aliases']

        # get index of all eligible devices
        sleep_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(sleep_device_types)) &
                                                  (device_info_copy['device_location'].isin(sleep_locations))].index.values.tolist()

        # if multiple eligible devices we will try to choose one
        if len(sleep_device_index) > 1:

            # if dominant hand is info is available we will choose based on dominant argument
            if coll.subject_info['dominant_hand'] in ['right', 'left']:

                # select dominant or non-dominant based on argument
                if dominant:
                    wrist = 'rwrist' if coll.subject_info['dominant_hand'] == 'right' else 'lwrist'
                else:
                    wrist = 'lwrist' if coll.subject_info['dominant_hand'] == 'right' else 'rwrist'

                # select devices at locations based on dominance
                sleep_locations = self.device_locations[wrist]['aliases']
                sleep_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(sleep_device_types)) &
                                                          (device_info_copy['device_location'].isin(sleep_locations))].index.values.tolist()

                # if still multiple eligible devices, take first one
                if len(sleep_device_index) > 1:
                    sleep_device_index = [sleep_device_index[0]]

                # if no eligible devices, go back and take first one from list of all eligible
                elif len(sleep_device_index) < 1:
                    sleep_locations = self.device_locations['rwrist']['aliases'] + self.device_locations['lwrist']['aliases']
                    sleep_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(sleep_device_types)) &
                                                              (device_info_copy['device_location'].isin(sleep_locations))].index.values.tolist()
                    sleep_device_index = [sleep_device_index[0]]

            # if no dominant hand info take first from list
            else:
                sleep_device_index = [sleep_device_index[0]]

        # if only one device determine, if it is dominant
        elif len(sleep_device_index) == 1:

            # if dominant hand info is available we will determine dominance
            if coll.subject_info['dominant_hand'] in ['right', 'left']:
                dominant_wrist = coll.subject_info['dominant_hand'][0] + 'wrist'
                dominant = device_info_copy.loc[sleep_device_index]['device_location'].item() in self.device_locations[dominant_wrist]['aliases']

            # if no dominant hand info available, assume dominant argument is correct

        return sleep_device_index, dominant

    def identify_df(self, coll, df):
        df.insert(loc=0, column='study_code', value=self.study_code)
        df.insert(loc=1, column='subject_id', value=coll.subject_id)
        df.insert(loc=2, column='coll_id', value=coll.coll_id)
        return df

    def get_collections(self):

        collections = [(row['subject_id'], row['coll_id']) for i, row in self.device_info.iterrows()]

        collections = list(set(collections))
        collections.sort()

        return collections

    def get_subject_ids(self):

        subject_ids = self.device_info['subject_id'].unique()
        subject_ids.sort()

        return subject_ids

    def get_coll_ids(self):

        coll_ids = self.device_info['coll_id'].unique()
        coll_ids.sort()

        return coll_ids



class Collection:

    def __init__(self, study_code, subject_id, coll_id):

        self.study_code = study_code
        self.subject_id = subject_id
        self.coll_id = coll_id

        self.devices = []


def message(msg, level='info', display=True, log=True, logger_name=None):

    level_switch = {'debug': lambda: logger.debug(msg),
                    'info': lambda: logger.info(msg),
                    'warning': lambda: logger.warning(msg),
                    'error': lambda: logger.error(msg),
                    'critical': lambda: logger.critical(msg)}

    logger = logging.getLogger(logger_name)

    if display:
        print(msg)

    if log:
        func = level_switch.get(level, lambda: 'Invalid')
        func()


class NWException(Exception):
    """Hit NWException when an expected error occurs in pipeline"""
    pass