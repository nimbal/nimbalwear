import os
import shutil
import datetime as dt
from pathlib import Path
import logging
import traceback
from functools import wraps
import json
import operator

from tqdm import tqdm
import numpy as np
import pandas as pd
from isodate import parse_duration

from .data import Device
from .nonwear import vert_nonwear, nonwear_stats
from .sleep import detect_sptw, detect_sleep_bouts, sptw_stats
from .gait import AccelReader, WalkingBouts, get_gait_bouts, create_timestamps, gait_stats
from .activity import activity_wrist_avm, activity_stats

from .__version__ import __version__


class Pipeline:

    def __init__(self, study_dir, settings_path=None):

        self.quiet = False
        self.log = True

        # initialize folder structure
        self.study_dir = Path(study_dir)

        self.settings_path = Path(settings_path) if settings_path is not None else settings_path

        if (self.settings_path is None) or (not self.settings_path.is_file()):
            self.settings_path = self.study_dir / 'pipeline/settings/settings.json'
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.settings_path.is_file():
            settings_src = Path(__file__).parent.absolute() / 'settings/settings.json'
            shutil.copy(settings_src, self.settings_path)

        # get study code
        self.study_code = self.study_dir.name

        # read json file
        with open(self.settings_path, 'r') as f:
            settings_json = json.load(f)

        self.dirs = settings_json['pipeline']['dirs']
        self.dirs = {key: self.study_dir / value for key, value in self.dirs.items()}

        # pipeline data files
        self.device_info_path = self.dirs['pipeline'] / 'devices.csv'
        self.subject_info_path = self.dirs['pipeline'] / 'subjects.csv'
        # self.log_file_path = self.dirs['logs'] / 'processing.log'
        self.status_path = self.dirs['pipeline'] / 'status.csv'

        self.stages = settings_json['pipeline']['stages']
        self.sensors = settings_json['pipeline']['sensors']
        self.device_locations = settings_json['pipeline']['device_locations']
        self.module_settings = settings_json['modules']

        self.settings_str = json.dumps(settings_json, indent=4)

        with open(Path(__file__).parent.absolute() / 'settings/data_dicts.json', 'r') as f:
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

        return

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

    def run(self, collections=None, single_stage=None, quiet=False, log=True, log_level=logging.INFO):
        """

        :param collections: list of tuples (subject_id, coll_id), default is None which will run all collections
        :param single_stage:

        :return:

        """

        self.quiet = quiet
        self.log = log

        # get all unique collections if none provided
        collections = self.get_collections() if collections is None else collections

        # TODO: ensure collections is a list of tuples

        for collection in tqdm(collections, desc="Processing collections", leave=True):

            subject_id = collection[0]
            coll_id = collection[1]

            self.log_name = f'{subject_id}_{coll_id}_{dt.datetime.now().strftime("%Y%m%d%H%M%S")}'
            log_path = self.dirs['logs'] / (self.log_name + '.log')

            fileh = logging.FileHandler(log_path, 'a')
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            fileh.setFormatter(formatter)
            fileh.setLevel(log_level)

            logger = logging.getLogger(self.log_name)
            for hdlr in logger.handlers[:]:  # remove all old handlers
                logger.removeHandler(hdlr)
            logger.setLevel(log_level)
            logger.addHandler(fileh)

            message("\n\n", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message(f"---- Processing collection ----------------------------------------------",
                    level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message(f"---- Study {self.study_code}, Subject {subject_id}, Collection {coll_id} --------", level='info', display=(not self.quiet),
                    log=self.log, logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)

            message(f"nimbalwear v{__version__}", level='info', display=(not self.quiet), log=self.log,
                    logger_name=self.log_name)

            if single_stage is not None:
                message(f"Single stage: {single_stage}", level='info', display=(not self.quiet), log=self.log,
                        logger_name=self.log_name)
            if not isinstance(self.subject_info, pd.DataFrame):
                message("Missing subjects info file in meta folder `subjects.csv`", level='warning',
                        display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message(f"Settings: {self.settings_path}\n\n {self.settings_str}", level='info', display=(not self.quiet),
                    log=self.log, logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)

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
                message(tb, level='error', display=(not self.quiet), log=self.log, logger_name=self.log_name)

            del coll

            message("---- End ----------------------------------------------\n", level='info', display=(not self.quiet),
                    log=self.log, logger_name=self.log_name)

        return

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
            coll = self.required_devices(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        # read data from all devices in collection
        coll = self.read(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        # convert to edf
        if single_stage in [None, 'convert']:
            coll = self.convert(coll=coll, quiet=self.quiet, log=self.log)

        # data integrity ??

        # process nonwear for all devices
        if single_stage in [None, 'nonwear']:
            coll = self.nonwear(coll=coll, quiet=self.quiet, log=self.log)

        if single_stage in ['crop', 'sleep', 'activity']:
            coll = self.read_nonwear(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        if single_stage in ['activity']:
            coll = self.read_sleep(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        if single_stage in ['activity']:
            coll = self.read_gait(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        # crop final nonwear
        if single_stage in [None, 'crop']:
            coll = self.crop(coll=coll, quiet=self.quiet, log=self.log)

        # save sensor edf files
        if single_stage in [None, 'save_sensors']:
            coll = self.save_sensors(coll=coll, quiet=self.quiet, log=self.log)

        # process posture

        # process gait
        if single_stage in [None, 'gait']:
            coll = self.gait(coll=coll, quiet=self.quiet, log=self.log, )

        # process sleep
        if single_stage in [None, 'sleep']:
            coll = self.sleep(coll=coll, quiet=self.quiet, log=self.log)

        # process activity levels
        if single_stage in [None, 'activity']:
            coll = self.activity(coll=coll, quiet=self.quiet, log=self.log)

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
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        overwrite_header = self.module_settings['read']['overwrite_header']

        # TODO: move to json or make autodetect?
        import_switch = {'EDF': lambda: device_data.import_edf(device_file_path, quiet=quiet),
                         'GNOR': lambda: device_data.import_geneactiv(device_file_path, correct_drift=True,
                                                                      quiet=quiet),
                         'AXV6': lambda: device_data.import_axivity(device_file_path, resample=True, quiet=quiet),
                         'BF18': lambda: device_data.import_bittium(device_file_path, quiet=quiet),
                         'BF36': lambda: device_data.import_bittium(device_file_path, quiet=quiet),
                         'NOWO': lambda: device_data.import_nonin(device_file_path, quiet=quiet)}


        # initialize list of collection device objects
        coll.devices = []

        # initialize list of device objects to be removed if file does not exist
        remove_idx = []

        # read in all data files for one collection
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

                #TODO: Rotate GENEActiv 90 deg if location is ankle

            elif single_stage in ['nonwear', 'crop']:

                device_file_path = self.dirs['device_edf_standard'] / device_edf_name
                import_func = import_switch.get('EDF', lambda: 'Invalid')

            else:

                device_file_path = self.dirs['device_edf_cropped'] / device_edf_name
                import_func = import_switch.get('EDF', lambda: 'Invalid')

            # check that data file exists
            if not device_file_path.exists():

                # if file does not exist then log,
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: {device_file_path} does not exist - "
                        + "this device will be excluded from further processing",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

                # store list of device_info rows to be removed,
                remove_idx.append(index)

                # go to next row
                continue

            # import data to device data object
            message(f"Reading {device_file_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            device_data = Device()
            import_func()
            device_data.deidentify()

            mismatch = False

            # check header against device list info
            header_comp = {'study_code': [(device_data.header['study_code'] == study_code),
                                          device_data.header['study_code'],
                                          coll.study_code],
                           'subject_id': [(device_data.header['subject_id'] == subject_id),
                                          device_data.header['subject_id'],
                                          subject_id],
                           'coll_id': [(device_data.header['coll_id'] == coll_id),
                                       device_data.header['coll_id'],
                                       coll_id],
                           'device_type': [(device_data.header['device_type'] == device_type),
                                           device_data.header['device_type'],
                                           device_type],
                           'device_id': [(device_data.header['device_id'] == device_id),
                                         device_data.header['device_id'],
                                         device_id],
                           'device_location': [(device_data.header['device_location'] == device_location),
                                               device_data.header['device_location'],
                                               device_location]}

            # generate message if any mismatches
            for key, value in header_comp.items():
                if not value[0]:
                    message(f"{subject_id}_{coll_id}_{device_type}_{device_location}:  {key} mismatch: " +
                            f"{value[1]} (header) != {value[2]} (device list)",
                            level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                    mismatch = True

            if mismatch and overwrite_header:

                message("Overwriting header from device list", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)

                device_data.header['study_code'] = study_code
                device_data.header['subject_id'] = subject_id
                device_data.header['coll_id'] = coll_id
                device_data.header['device_type'] = device_type
                device_data.header['device_id'] = device_id
                device_data.header['device_location'] = device_location

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

            coll.devices.append(device_data)

        #remove devices from device_info if file was not found
        coll.device_info = coll.device_info.drop(index=remove_idx).reset_index(drop=True)

        return coll

    @coll_status
    def convert(self, coll, quiet=False, log=True):

        if self.module_settings['convert']['autocal']:
            coll = self.autocal(coll, quiet=quiet, log=log)

        if self.module_settings['convert']['sync']:
            coll = self.sync(coll, quiet=quiet, log=log)

        if self.module_settings['convert']['adj_start']:
            coll = self.adj_start(coll, quiet=quiet, log=log)

        message("Converting device data to EDF...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        # save all device data to edf
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
                    logger_name=self.log_name)

            # write device data as edf
            coll.devices[index].export_edf(file_path=standard_device_path, quiet=quiet)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def autocal(self, coll, quiet=False, log=True):

        message("Autocalibrating device data...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        #TODO: only calibrate devices with Accelerometer and only use_temp if Temperature signal exists

        for idx, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0],
                             desc="Autocalibrating devices", leave=False):
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_id = row['device_id']

            pre_err, post_err, iter = coll.devices[idx].autocal(quiet=quiet)

            if pre_err is None:
                message(f"Autocalibration for {device_type} {device_location} could not be performed.",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            calib = pd.DataFrame({'study_code': study_code,
                                  'subject_id': subject_id,
                                  'coll_id': coll_id,
                                  'device_type': device_type,
                                  'device_location': device_location,
                                  'device_id': device_id,
                                  'pre_err': pre_err,
                                  'post_err': post_err,
                                  'iter': iter}, index=[0])

            message(f"Autocalibrated {device_type} {device_location}: Calibration error reduced from {pre_err} to {post_err} after {iter} iterations.",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            if self.module_settings['autocal']['save']:

                # create all file path variables
                calib_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                     "CALIB"]),
                                           "csv"])

                calib_csv_path = self.dirs['calib'] / calib_csv_name


                calib_csv_path.parent.mkdir(parents=True, exist_ok=True)

                message(f"Saving {calib_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)
                calib.to_csv(calib_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def sync(self, coll, quiet=False, log=True):

        message("Synchronizing device data...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        sync_type = self.module_settings['sync']['type']
        sync_at_config = self.module_settings['sync']['sync_at_config']
        search_radius = self.module_settings['sync']['search_radius']

        if not coll.device_info.empty:
            ref_device_type = coll.device_info.iloc[0]['device_type']
            ref_device_location = coll.device_info.iloc[0]['device_location']

        for idx, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0],
                             desc="Synchronizing devices", leave=False):
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']

            if idx == 0:

                # check if sync_at_config is true and give warning and set to false if config_date after start_date
                if (sync_at_config) & (coll.devices[idx].header['config_datetime'] > coll.devices[idx].header['start_datetime']):
                    sync_at_config = False
                    message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: Invalid config time, could not add as sync time",
                            level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                    message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

            else:

                accel_idx = coll.devices[idx].get_signal_index(self.sensors['accelerometer']['signals'][0])

                # set signal_ds to downsample to somewhere between 5-11 Hz for sync detection if possible
                freq = coll.devices[idx].signal_headers[accel_idx]['sample_rate']
                try:
                    ds_index = [freq % x for x in range(5, 12)].index(0)
                except ValueError:
                    ds_index = freq - 5
                signal_ds = round(freq / (5 + ds_index))

                syncs, segments = coll.devices[idx].sync(ref=coll.devices[0],
                                                         sig_labels=tuple(self.sensors['accelerometer']['signals']),
                                                         type=sync_type,
                                                         sync_at_config=sync_at_config,
                                                         search_radius=search_radius,
                                                         signal_ds=signal_ds)


                message(f"Synchronized {device_type} {device_location} to {ref_device_type} {ref_device_location} at {syncs.shape[0]} sync points",
                        level='info', display=(not quiet), log=log, logger_name=self.log_name)

                syncs.insert(loc=0, column='study_code', value=study_code)
                syncs.insert(loc=1, column='subject_id', value=subject_id)
                syncs.insert(loc=2, column='coll_id', value=coll_id)
                syncs.insert(loc=3, column='device_type', value=device_type)
                syncs.insert(loc=4, column='device_location', value=device_location)
                syncs.insert(loc=5, column='sync_id', value=range(1, syncs.shape[0] + 1))

                segments.insert(loc=0, column='study_code', value=study_code)
                segments.insert(loc=1, column='subject_id', value=subject_id)
                segments.insert(loc=2, column='coll_id', value=coll_id)
                segments.insert(loc=3, column='device_type', value=device_type)
                segments.insert(loc=4, column='device_location', value=device_location)
                segments.insert(loc=5, column='segment_id', value=range(1, segments.shape[0] + 1))

                if self.module_settings['sync']['save']:

                    # create all file path variables
                    syncs_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                         "SYNC_LOC"]),
                                               "csv"])
                    segments_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                            "SYNC_SEG"]),
                                                  "csv"])

                    syncs_csv_path = self.dirs['sync'] / syncs_csv_name
                    segments_csv_path = self.dirs['sync'] / segments_csv_name

                    syncs_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    segments_csv_path.parent.mkdir(parents=True, exist_ok=True)

                    message(f"Saving {syncs_csv_path}", level='info', display=(not quiet), log=log,
                            logger_name=self.log_name)
                    syncs.to_csv(syncs_csv_path, index=False)

                    message(f"Saving {segments_csv_path}", level='info', display=(not quiet), log=log,
                            logger_name=self.log_name)
                    segments.to_csv(segments_csv_path, index=False)

                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def adj_start(self, coll, quiet=False, log=True):

        # TODO: determine if config_datetime should also be adjusted

        message("Adjusting device start times...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        # duration is stored in json in iso 8601 format
        duration_iso = self.module_settings['convert']['adj_start']

        # default to add if no operator specified
        op = operator.add

        # if operator is specified then isolate from duration
        if duration_iso[0] in ["+", "-"]:
            ops = {"+": operator.add,
                   "-": operator.sub}
            op = ops[duration_iso[0]]
            duration_iso = duration_iso[1:]

        # convert iso duration to timedelta
        duration_delta = parse_duration(duration_iso)

        # adjust start_datetime for each device
        for idx, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Adjusting device start times'):

            device_type = row['device_type']
            device_location = row['device_location']

            old_start_datetime = coll.devices[idx].header['start_datetime']
            new_start_datetime = op(old_start_datetime, duration_delta)

            coll.devices[idx].header['start_datetime'] = new_start_datetime

            message(f"Adjusted {device_type} {device_location} start time from {old_start_datetime} to {new_start_datetime}",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def nonwear(self, coll, quiet=False, log=True):

        # process nonwear for all devices
        message("Detecting non-wear...", level='info', display=(not quiet), log=log, logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        accel_std_thresh_mg = self.module_settings['nonwear']['accel_std_thresh_mg']
        low_temperature_cutoff = self.module_settings['nonwear']['low_temperature_cutoff']
        high_temperature_cutoff = self.module_settings['nonwear']['high_temperature_cutoff']
        temp_dec_roc = self.module_settings['nonwear']['temp_dec_roc']
        temp_inc_roc = self.module_settings['nonwear']['temp_inc_roc']
        save = self.module_settings['nonwear']['save']

        coll.nonwear_bouts = pd.DataFrame()
        coll.daily_nonwear = pd.DataFrame()

        # detect nonwear for each device
        for i, r in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Detecting non-wear'):

            # get info from device list
            study_code = r['study_code']
            subject_id = r['subject_id']
            coll_id = r['coll_id']
            device_type = r['device_type']
            device_location = r['device_location']

            device = coll.devices[i]

            # TODO: Add nonwear detection for other devices
            # TODO: add wear bouts
            # TODO: rename from nonwear to wear

            if device_type not in ['AXV6', 'GNOR']:
                message(f"Cannot detect non-wear for {device_type}_{device_location}",
                        level='info', display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            # check for data loaded
            if device is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            accel_x_idx = device.get_signal_index('Accelerometer x')
            accel_y_idx = device.get_signal_index('Accelerometer y')
            accel_z_idx = device.get_signal_index('Accelerometer z')
            temperature_idx = device.get_signal_index('Temperature')

            # TODO: call different algorithm based on device_type or signals available??
            # TODO: log algorithm used

            accel_x = device.signals[accel_x_idx]
            accel_y = device.signals[accel_y_idx]
            accel_z = device.signals[accel_z_idx]
            temperature = device.signals[temperature_idx]

            # TODO: index signals by label - make option to return datetimeindex

            accel_fs = device.signal_headers[accel_x_idx]['sample_rate']
            temperature_fs = device.signal_headers[temperature_idx]['sample_rate']


            nonwear_bouts, nonwear_array = vert_nonwear(x_values=accel_x, y_values=accel_y, z_values=accel_z,
                                                        temperature_values=temperature, accel_freq=accel_fs,
                                                        temperature_freq=temperature_fs,
                                                        std_thresh_mg=accel_std_thresh_mg,
                                                        low_temperature_cutoff=low_temperature_cutoff,
                                                        high_temperature_cutoff=high_temperature_cutoff,
                                                        temp_dec_roc=temp_dec_roc, temp_inc_roc=temp_inc_roc,
                                                        quiet=quiet)

            algorithm_name = 'DETACH'


            nonwear_bouts['event'] = "nonwear"
            nonwear_bouts.rename(columns={'Start Datapoint': 'start_datapoint', 'End Datapoint': 'end_datapoint'},
                                 inplace=True)

            bout_count = nonwear_bouts.shape[0]

            message(f"Detected {bout_count} nonwear bouts for {device_type} {device_location} ({algorithm_name})",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            # convert datapoints to times
            start_date = device.header['start_datetime']
            sample_rate = device.signal_headers[accel_x_idx]['sample_rate']
            samples = device.signals[accel_x_idx].shape[0]
            end_date = start_date + dt.timedelta(seconds=(samples / sample_rate))

            nonwear_start_times = []
            nonwear_end_times = []

            for nw_index, nw_row in nonwear_bouts.iterrows():
                nonwear_start_times.append(start_date + dt.timedelta(seconds=(nw_row['start_datapoint'] / sample_rate)))
                nonwear_end_times.append(start_date + dt.timedelta(seconds=(nw_row['end_datapoint'] / sample_rate)))

            nonwear_bouts['start_time'] = nonwear_start_times
            nonwear_bouts['end_time'] = nonwear_end_times

            nonwear_bouts = nonwear_bouts[['event', 'start_time', 'end_time']]

            # nonwear end times are wear start times -- nonwear start times are wear end times
            wear_start_times = nonwear_end_times
            wear_end_times = nonwear_start_times

            # collection start is first wear start
            wear_start_times.insert(0, start_date)

            # collection end is last wear end
            wear_end_times.append(end_date)

            # remove first wear bout if duration is 0 - started with non-wear
            if wear_start_times[0] == wear_end_times[0]:
                wear_start_times = wear_start_times[1:]
                wear_end_times = wear_end_times[1:]

            if wear_start_times[-1] == wear_end_times[-1]:
                wear_start_times = wear_start_times[:-1]
                wear_end_times = wear_end_times[:-1]

            wear_bouts = pd.DataFrame({'start_time': wear_start_times, 'end_time': wear_end_times, })
            wear_bouts['event'] = 'wear'

            nonwear_bouts = pd.concat([nonwear_bouts, wear_bouts], ignore_index=True)
            nonwear_bouts = nonwear_bouts.sort_values('start_time')

            nonwear_bouts.insert(loc=0, column='id', value=range(1, nonwear_bouts.shape[0] + 1))
            daily_nonwear = nonwear_stats(nonwear_bouts, quiet=quiet)

            # add identifiers
            nonwear_bouts.insert(loc=0, column='study_code', value=study_code)
            nonwear_bouts.insert(loc=1, column='subject_id', value=subject_id)
            nonwear_bouts.insert(loc=2, column='coll_id', value=coll_id)
            nonwear_bouts.insert(loc=3, column='device_type', value=device_type)
            nonwear_bouts.insert(loc=4, column='device_location', value=device_location)

            daily_nonwear.insert(loc=0, column='study_code', value=study_code)
            daily_nonwear.insert(loc=1, column='subject_id', value=subject_id)
            daily_nonwear.insert(loc=2, column='coll_id', value=coll_id)
            daily_nonwear.insert(loc=3, column='device_type', value=device_type)
            daily_nonwear.insert(loc=4, column='device_location', value=device_location)


            # reorder columns
            # nonwear_bouts = nonwear_bouts[['study_code', 'subject_id', 'coll_id', 'device_type', 'device_location',
            #                               'id', 'event', 'start_time', 'end_time']]

            # append to collection attribute
            coll.nonwear_bouts = pd.concat([coll.nonwear_bouts, nonwear_bouts], ignore_index=True)
            coll.daily_nonwear = pd.concat([coll.daily_nonwear, daily_nonwear], ignore_index=True)

            if save:

                # create all file path variables
                nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                       "NONWEAR"]),
                                             "csv"])
                daily_nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                       "NONWEAR_DAILY"]),
                                             "csv"])

                nonwear_csv_path = self.dirs['nonwear_bouts_standard'] / nonwear_csv_name
                nonwear_daily_csv_path = self.dirs['nonwear_daily_standard'] / daily_nonwear_csv_name

                nonwear_csv_path.parent.mkdir(parents=True, exist_ok=True)
                nonwear_daily_csv_path.parent.mkdir(parents=True, exist_ok=True)

                message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)

                nonwear_bouts.to_csv(nonwear_csv_path, index=False)

                message(f"Saving {nonwear_daily_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)

                daily_nonwear.to_csv(nonwear_daily_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def read_nonwear(self, coll, single_stage, quiet=False, log=True):

        # read nonwear data for all devices
        message("Reading non-wear data from files...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        if single_stage == 'crop':
            nonwear_csv_dir = self.dirs['nonwear_bouts_standard']
        else:
            nonwear_csv_dir = self.dirs['nonwear_bouts_cropped']

        coll.nonwear_bouts = pd.DataFrame()

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
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
                #coll.devices.append(None)    THIS SHOULD NOT BE HERE? CUT AND PASTE ERROR?
                continue

            message(f"Reading {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            # read nonwear csv file
            nonwear_bouts = pd.read_csv(nonwear_csv_path, dtype=str)
            nonwear_bouts['start_time'] = pd.to_datetime(nonwear_bouts['start_time'], format='%Y-%m-%d %H:%M:%S')
            nonwear_bouts['end_time'] = pd.to_datetime(nonwear_bouts['end_time'], format='%Y-%m-%d %H:%M:%S')

            # append to collection attribute
            coll.nonwear_bouts = pd.concat([coll.nonwear_bouts, nonwear_bouts], ignore_index=True)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def crop(self, coll, quiet=False, log=True):

        message("Cropping initial and final non-wear...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        # min_duration_start = self.module_settings['crop']['min_duration_start']
        # min_duration_end = self.module_settings['crop']['min_duration_end']
        # max_time_to_bof = self.module_settings['crop']['max_time_to_bof']
        # max_time_to_eof = self.module_settings['crop']['max_time_to_eof']
        min_wear_time = self.module_settings['crop']['min_wear_time']
        save = self.module_settings['crop']['save']

        nonwear_bouts = coll.nonwear_bouts.copy()
        nonwear_bouts['duration'] = ((nonwear_bouts['end_time'] - nonwear_bouts['start_time']).dt.total_seconds() / 60).round()

        coll.daily_nonwear = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'device_type',
                                                   'device_location', 'day_num', 'date', 'wear', 'nonwear'])

        coll.nonwear_bouts = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'device_type',
                                                   'device_location', 'id', 'event', 'start_time', 'end_time'])



        # crop final nonwear from all device data
        for i, r in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Cropping initial and final non-wear'):

            # get info from device list
            study_code = r['study_code']
            subject_id = r['subject_id']
            coll_id = r['coll_id']
            device_type = r['device_type']
            device_location = r['device_location']

            device = coll.devices[i]

            if device is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            # if there is nonwear data for any devices in this collection
            if not nonwear_bouts.empty:

                daily_nonwear = pd.DataFrame(columns=['day_num', 'date', 'wear', 'nonwear'])

                # get nonwear bouts for current device
                device_bouts = nonwear_bouts.loc[(nonwear_bouts['study_code'] == study_code) &
                                                 (nonwear_bouts['subject_id'] == subject_id) &
                                                 (nonwear_bouts['coll_id'] == coll_id) &
                                                 (nonwear_bouts['device_type'] == device_type) &
                                                 (nonwear_bouts['device_location'] == device_location)].copy()

                if not device_bouts.empty:

                    long_wear_idxs = device_bouts.index[(device_bouts['event'] == 'wear')
                                                    & (device_bouts['duration'] > min_wear_time)]

                    if not long_wear_idxs.empty:

                        device_bouts = device_bouts.loc[long_wear_idxs[0]:long_wear_idxs[-1]]

                        if not device_bouts.empty:

                        # nonwear_idx = nonwear_idx.tolist()
                        #
                        # # if there is nonwear data for current device
                        # if len(nonwear_idx):
                        #
                        #     # get first nonwear period for current device
                        #     first_nonwear_idx = nonwear_idx[0]
                        #     first_nonwear = nonwear_bouts_keep.loc[first_nonwear_idx]
                        #
                        #     # get last nonwear period for current device
                        #     last_nonwear_idx = nonwear_idx[-1]
                        #     last_nonwear = nonwear_bouts_keep.loc[last_nonwear_idx]
                        #
                            # get time info from device data
                            start_time = device.header['start_datetime']
                            samples = len(device.signals[0])
                            sample_rate = device.signal_headers[0]['sample_rate']
                            duration = dt.timedelta(seconds=samples / sample_rate)
                            end_time = start_time + duration
                        #
                        #     # get duration and time to start of file of first nonwear
                        #     first_nonwear_duration = first_nonwear['end_time'] - first_nonwear['start_time']
                        #     first_nonwear_time_to_bof = first_nonwear['start_time'] - start_time
                        #
                        #     # get duration and time to end of file of last nonwear
                        #     last_nonwear_duration = last_nonwear['end_time'] - last_nonwear['start_time']
                        #     last_nonwear_time_to_eof = end_time - last_nonwear['end_time']
                        #
                        #     # only crop if first nonwear starts within 20 minutes of start of file
                        #     crop_start = ((first_nonwear_duration >= dt.timedelta(minutes=min_duration_end)) &
                        #                 (first_nonwear_time_to_bof <= dt.timedelta(minutes=max_time_to_bof)))
                        #
                        #     # only crop if last nonwear ends within 20 minutes of end of file
                        #     crop_end = ((last_nonwear_duration >= dt.timedelta(minutes=min_duration_end)) &
                        #                     (last_nonwear_time_to_eof <= dt.timedelta(minutes=max_time_to_eof)))

                            new_start_time = device_bouts.iloc[0]['start_time']
                            new_end_time = device_bouts.iloc[-1]['end_time']

                            start_crop_duration = new_start_time - start_time

                            message(f"Cropping {start_crop_duration} from begininng of collection for {device_type} {device_location}",
                                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

                            end_crop_duration = end_time - new_end_time

                            message(f"Cropping {end_crop_duration} from end of collection for {device_type} {device_location}",
                                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

                            device.crop(new_start_time, new_end_time, inplace=True)

                            # recalculate nonwear summary
                            #nonwear_bouts =  nonwear_bouts_keep[nonwear_bouts_keep.index.isin(nonwear_idx)]
                            db = device_bouts.drop(columns=['study_code', 'subject_id', 'coll_id', 'device_type',
                                                                     'device_location', 'duration'], )
                            daily_nonwear = nonwear_stats(db, quiet=quiet)

                    else:
                        message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: Could not crop due to lack of wear time",
                                level='warning', display=(not quiet), log=log, logger_name=self.log_name)

                else:
                    message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No nonwear data for device",
                            level='warning', display=(not quiet), log=log, logger_name=self.log_name)

                daily_nonwear.insert(loc=0, column='study_code', value=study_code)
                daily_nonwear.insert(loc=1, column='subject_id', value=subject_id)
                daily_nonwear.insert(loc=2, column='coll_id', value=coll_id)
                daily_nonwear.insert(loc=3, column='device_type', value=device_type)
                daily_nonwear.insert(loc=4, column='device_location', value=device_location)

                coll.daily_nonwear = pd.concat([coll.daily_nonwear, daily_nonwear], ignore_index=True)
                device_bouts = device_bouts.drop(columns=['duration'])
                coll.nonwear_bouts = pd.concat([coll.nonwear_bouts, device_bouts], ignore_index=True)
                coll.devices[i] = device

            else:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No nonwear data for collection",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)

            if save:

                # create all file path variables
                device_edf_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location]),
                                            "edf"])
                nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                       "NONWEAR"]),
                                             "csv"])
                daily_nonwear_csv_name = '.'.join(['_'.join([study_code, subject_id, coll_id, device_type, device_location,
                                                       "NONWEAR_DAILY"]),
                                             "csv"])

                cropped_device_path = self.dirs['device_edf_cropped'] / device_edf_name
                nonwear_csv_path = self.dirs['nonwear_bouts_cropped'] / nonwear_csv_name
                nonwear_daily_csv_path = self.dirs['nonwear_daily_cropped'] / daily_nonwear_csv_name

                # check that all folders exist for data output files
                cropped_device_path.parent.mkdir(parents=True, exist_ok=True)
                nonwear_csv_path.parent.mkdir(parents=True, exist_ok=True)
                nonwear_daily_csv_path.parent.mkdir(parents=True, exist_ok=True)

                # write cropped device data as edf
                message(f"Saving {cropped_device_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)
                device.export_edf(file_path=cropped_device_path, quiet=quiet)

                if not coll.nonwear_bouts.empty:

                    # write nonwear times with cropped nonwear removed
                    message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                            logger_name=self.log_name)
                    device_bouts.to_csv(nonwear_csv_path, index=False)

                    message(f"Saving {nonwear_daily_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)

                    daily_nonwear.to_csv(nonwear_daily_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def save_sensors(self, coll, quiet=False, log=True):

        message("Separating sensors from devices...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

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
                            logger_name=self.log_name)

                    coll.devices[index].export_edf(file_path=sensor_path, sig_nums_out=sig_nums, quiet=quiet)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def gait(self, coll, quiet=False, log=True):

        # TODO: axis needs to be set based on orientation of device

        step_detect_type = self.module_settings['gait']['step_detect_type']
        axis = self.module_settings['gait']['axis']
        save = self.module_settings['gait']['save']

        message(f"Detecting steps and walking bouts using {step_detect_type} data...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        r_gait_device_index, l_gait_device_index = self.select_gait_device(coll=coll)

        if not (l_gait_device_index or r_gait_device_index):
            raise NWException(f'{coll.subject_id}_{coll.coll_id}: No left or right ankle device found in device list')


        if step_detect_type == 'accel':

            #######################
            # ACCEL GAIT DETECT
            #######################

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

            l_obj = AccelReader.sig_init(raw_x=coll.devices[l_gait_device_index].signals[l_accel_x_sig],
                raw_y=coll.devices[l_gait_device_index].signals[l_accel_y_sig],
                raw_z=coll.devices[l_gait_device_index].signals[l_accel_z_sig],
                startdate = coll.devices[l_gait_device_index].header['start_datetime'],
                freq=coll.devices[l_gait_device_index].signal_headers[l_accel_x_sig]['sample_rate'])

            r_accel_x_sig = coll.devices[r_gait_device_index].get_signal_index('Accelerometer x')
            r_accel_y_sig = coll.devices[r_gait_device_index].get_signal_index('Accelerometer y')
            r_accel_z_sig = coll.devices[r_gait_device_index].get_signal_index('Accelerometer z')

            r_obj = AccelReader.sig_init(raw_x=coll.devices[r_gait_device_index].signals[r_accel_x_sig],
                raw_y=coll.devices[r_gait_device_index].signals[r_accel_y_sig],
                raw_z=coll.devices[r_gait_device_index].signals[r_accel_z_sig],
                startdate = coll.devices[r_gait_device_index].header['start_datetime'],
                freq=coll.devices[r_gait_device_index].signal_headers[r_accel_x_sig]['sample_rate'])

            # run gait algorithm to find bouts
            # TODO: Add progress bars instead of print statements??
            wb = WalkingBouts(l_obj, r_obj, left_kwargs={'axis': axis}, right_kwargs={'axis': axis})

            # save bout times
            coll.gait_bouts = wb.export_bouts()

            # save step times
            coll.gait_step_times = wb.export_steps()

            # compensate for export_steps returning blank DataFrame if no steps
            # TODO: Fix in nwgait to return columns
            if coll.gait_step_times.empty:
                coll.gait_step_times = pd.DataFrame(columns=['step_num', 'gait_bout_num', 'foot', 'avg_speed',
                                                        'heel_strike_accel', 'heel_strike_time', 'mid_swing_accel',
                                                        'mid_swing_time', 'step_length', 'step_state', 'step_time',
                                                        'swing_start_accel', 'swing_start_time'])

            coll.gait_bouts = self.identify_df(coll, coll.gait_bouts)
            coll.gait_step_times = self.identify_df(coll, coll.gait_step_times)

            message(f"Detected {coll.gait_bouts.shape[0]} gait bouts", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            message(f"Detected {coll.gait_step_times.shape[0]} steps",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            message("Summarizing daily gait analytics...", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            coll.gait_daily = gait_stats(coll.gait_bouts)
            coll.gait_daily = self.identify_df(coll, coll.gait_daily)

            # adjusting gait parameters
            coll.gait_bouts.rename(columns={'start_dp': 'start_idx',
                                            'end_dp': 'end_idx'},
                                    inplace=True)

            coll.gait_step_times.rename(columns={'step_index': 'step_idx'}, inplace=True)


        elif step_detect_type == 'gyro':

            #####################
            # Gyro Gait Detect
            #####################

            # TODO: currently works only for single leg - limitation of algorithm perhaps ???

            device_idx = r_gait_device_index if r_gait_device_index else l_gait_device_index
            device_idx = device_idx[0]

            gyro_z_idx = coll.devices[device_idx].get_signal_index("Gyroscope z")

            # creating timestamps && timestamp info if needed-----
            # start_stamp = file.header["start_datetime"]
            times, idxs = create_timestamps(data_start_time=coll.devices[device_idx].header["start_datetime"],
                                            data_len=len(coll.devices[device_idx].signals[gyro_z_idx]),
                                            fs=coll.devices[device_idx].signal_headers[gyro_z_idx]['sample_rate'])

            sgp = get_gait_bouts(data=coll.devices[device_idx].signals[gyro_z_idx],
                                 sample_freq=coll.devices[device_idx].signal_headers[gyro_z_idx]['sample_rate'],
                                 timestamps=times, break_sec=2, bout_steps=3, start_ind=idxs[0], end_ind=idxs[1])

            coll.gait_step_times, coll.gait_bouts, peak_heights = sgp

            # nnot sure if this is necessary in this version
            # TODO: Fix in nwgait to return columns
            if coll.gait_step_times.empty:
                coll.gait_step_times = pd.DataFrame(columns=['Step', 'Step_index', 'Bout_number', 'Peak_times'])

            coll.gait_bouts = self.identify_df(coll, coll.gait_bouts)
            coll.gait_step_times = self.identify_df(coll, coll.gait_step_times)

            message(f"Detected {coll.gait_bouts.shape[0]} gait bouts", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            message(f"Detected {coll.gait_step_times.shape[0] * 2} steps",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            message("Summarizing daily gait analytics...", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            # adjusting gait parameters
            coll.gait_bouts.rename(columns={'Bout_number': 'gait_bout_num',
                                                 'Step_count': 'step_count',
                                                 'Start_time': 'start_time',
                                                 'End_time': 'end_time',
                                                 'Start_idx': 'start_idx',
                                                 'End_idx': 'end_idx'},
                                        inplace=True)

            coll.gait_step_times.rename(columns={'Step': 'step_num',
                                                 'Step_index': 'step_idx',
                                                 'Bout_number': 'gait_bout_num',
                                                 'Peak_times': 'step_time'},
                                        inplace=True)

            coll.gait_daily = gait_stats(coll.gait_bouts, single_leg=True)
            coll.gait_daily = self.identify_df(coll, coll.gait_daily)




        else:
            message(f"Invalid step_detect_type: {step_detect_type}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            return coll


        bout_cols = ['study_code', 'subject_id', 'coll_id', 'gait_bout_num', 'start_time', 'end_time',
                     'step_count']
        coll.gait_bouts = coll.gait_bouts[bout_cols]


        step_cols = ['study_code','subject_id','coll_id','step_num', 'gait_bout_num', 'step_idx', 'step_time']
        coll.gait_step_times = coll.gait_step_times[step_cols]

        if save:
            # create all file path variables
            bouts_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_BOUTS"]), "csv"])
            steps_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_STEPS"]), "csv"])
            # steps_rej_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_STEPS_REJ"]),
            #                                "csv"])
            daily_gait_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_DAILY"]),
                                            "csv"])

            bouts_csv_path = self.dirs['gait_bouts'] / bouts_csv_name
            steps_csv_path = self.dirs['gait_steps'] / steps_csv_name
            # steps_rej_csv_path = self.dirs['gait_steps'] / steps_rej_csv_name
            daily_gait_csv_path = self.dirs['gait_daily'] / daily_gait_csv_name

            message(f"Saving {bouts_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.log_name)
            coll.gait_bouts.to_csv(bouts_csv_path, index=False)

            message(f"Saving {steps_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.log_name)
            coll.gait_step_times.to_csv(steps_csv_path, index=False)

            # message(f"Saving {steps_rej_csv_path}", level='info', display=(not quiet), log=log,
            #         logger_name=self.study_code)
            # coll.step_times[coll.step_times['step_state'] != 'success'].to_csv(steps_rej_csv_path, index=False)

            message(f"Saving {daily_gait_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.gait_daily.to_csv(daily_gait_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def read_gait(self, coll, single_stage, quiet=False, log=True):

        # read nonwear data for all devices
        message("Reading gait data from files...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        gait_bouts_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "GAIT_BOUTS"]),
                                         "csv"])

        gait_bouts_csv_path = self.dirs['gait_bouts'] / gait_bouts_csv_name

        coll.gait_bouts = pd.DataFrame()

        if os.path.isfile(gait_bouts_csv_path):

            message(f"Reading {gait_bouts_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            # read nonwear csv file
            coll.gait_bouts = pd.read_csv(gait_bouts_csv_path, dtype=str)
            coll.gait_bouts['start_time'] = pd.to_datetime(coll.gait_bouts['start_time'], format='%Y-%m-%d %H:%M:%S')
            coll.gait_bouts['end_time'] = pd.to_datetime(coll.gait_bouts['end_time'], format='%Y-%m-%d %H:%M:%S')

        else:
            message(f"{coll.subject_id}_{coll.coll_id}: {gait_bouts_csv_path} does not exist",
                    level='warning', display=(not quiet), log=log, logger_name=self.log_name)
            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def sleep(self, coll, quiet=False, log=True):

        message("Analyzing sleep...", level='info', display=(not quiet), log=log, logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

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
        device_nonwear = coll.nonwear_bouts.loc[(coll.nonwear_bouts['study_code'] == coll.study_code) &
                                                (coll.nonwear_bouts['subject_id'] == coll.subject_id) &
                                                (coll.nonwear_bouts['coll_id'] == coll.coll_id) &
                                                (coll.nonwear_bouts['device_type'] == coll.device_info.iloc[sleep_device_index]['device_type']) &
                                                (coll.nonwear_bouts['device_location'] == coll.device_info.iloc[sleep_device_index]['device_location']) &
                                                (coll.nonwear_bouts['event'] == 'nonwear')]

        # TODO: should sleep algorithm be modified if dominant vs non-dominant hand?

        coll.sptw, z_angle, z_angle_diff, z_sample_rate = detect_sptw(
            x_values=coll.devices[sleep_device_index].signals[accel_x_sig],
            y_values=coll.devices[sleep_device_index].signals[accel_y_sig],
            z_values=coll.devices[sleep_device_index].signals[accel_z_sig],
            sample_rate=round(coll.devices[sleep_device_index].signal_headers[accel_x_sig]['sample_rate']),
            start_datetime=coll.devices[sleep_device_index].header['start_datetime'],
            nonwear = device_nonwear)

        message(f"Detected {coll.sptw.shape[0]} sleep period time windows", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)

        sleep_t5a5 = detect_sleep_bouts(z_angle_diff=z_angle_diff, sptw=coll.sptw, z_sample_rate=z_sample_rate,
                                        start_datetime=coll.devices[sleep_device_index].header['start_datetime'],
                                        z_abs_threshold=5, min_sleep_length=5)

        sleep_t5a5.insert(loc=2, column='bout_detect', value='t5a5')

        message(f"Detected {sleep_t5a5.shape[0]} sleep bouts (t5a5)", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)

        sleep_t8a4 = detect_sleep_bouts(z_angle_diff=z_angle_diff, sptw=coll.sptw, z_sample_rate=z_sample_rate,
                                        start_datetime=coll.devices[sleep_device_index].header['start_datetime'],
                                        z_abs_threshold=4, min_sleep_length=8)

        sleep_t8a4.insert(loc=2, column='bout_detect', value='t8a4')

        message(f"Detected {sleep_t8a4.shape[0]} sleep bouts (t8a4)", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)

        coll.sleep_bouts = pd.concat([sleep_t5a5, sleep_t8a4])

        daily_sleep_t5a5 = sptw_stats(coll.sptw, sleep_t5a5, type='daily', sptw_inc=['long', 'all', 'sleep', 'overnight_sleep'])
        message(f"Summarized {daily_sleep_t5a5['sptw_inc'].value_counts()['long']} days of sleep analytics (t5a5)...",
                level='info', display=(not quiet), log=log, logger_name=self.log_name)

        daily_sleep_t8a4 = sptw_stats(coll.sptw, sleep_t8a4, type='daily', sptw_inc=['long', 'all', 'sleep', 'overnight_sleep'])
        message(f"Summarized {daily_sleep_t8a4['sptw_inc'].value_counts()['long']} days of sleep analytics (t8a4)...",
                level='info', display=(not quiet), log=log, logger_name=self.log_name)

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

            daily_sleep_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "SLEEP_DAILY"]),
                                             "csv"])

            sptw_csv_path = self.dirs['sleep_sptw'] / sptw_csv_name
            sleep_bouts_csv_path = self.dirs['sleep_bouts'] / sleep_bouts_csv_name
            daily_sleep_csv_path = self.dirs['sleep_daily'] / daily_sleep_csv_name

            sptw_csv_path.parent.mkdir(parents=True, exist_ok=True)
            sleep_bouts_csv_path.parent.mkdir(parents=True, exist_ok=True)
            daily_sleep_csv_path.parent.mkdir(parents=True, exist_ok=True)

            message(f"Saving {sptw_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.log_name)
            coll.sptw.to_csv(sptw_csv_path, index=False)

            message(f"Saving {sleep_bouts_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.sleep_bouts.to_csv(sleep_bouts_csv_path, index=False)

            message(f"Saving {daily_sleep_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.daily_sleep.to_csv(daily_sleep_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def read_sleep(self, coll, single_stage, quiet=False, log=True):

        # read nonwear data for all devices
        message("Reading sleep data from files...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        sptw_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "SPTW"]), "csv"])
        sleep_bouts_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id, coll.coll_id, "SLEEP_BOUTS"]),
                                         "csv"])

        sptw_csv_path = self.dirs['sleep_sptw'] / sptw_csv_name
        sleep_bouts_csv_path = self.dirs['sleep_bouts'] / sleep_bouts_csv_name

        coll.sptw = pd.DataFrame()
        coll.sleep_bouts = pd.DataFrame()

        if os.path.isfile(sptw_csv_path):

            message(f"Reading {sptw_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            # read nonwear csv file
            coll.sptw = pd.read_csv(sptw_csv_path, dtype=str)
            coll.sptw['start_time'] = pd.to_datetime(coll.sptw['start_time'], format='%Y-%m-%d %H:%M:%S')
            coll.sptw['end_time'] = pd.to_datetime(coll.sptw['end_time'], format='%Y-%m-%d %H:%M:%S')


        else:
            message(f"{coll.subject_id}_{coll.coll_id}: {sptw_csv_path} does not exist",
                    level='warning', display=(not quiet), log=log, logger_name=self.log_name)
            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        if os.path.isfile(sleep_bouts_csv_path):

            message(f"Reading {sleep_bouts_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            # read nonwear csv file
            coll.sleep_bouts = pd.read_csv(sleep_bouts_csv_path, dtype=str)
            coll.sleep_bouts['start_time'] = pd.to_datetime(coll.sleep_bouts['start_time'], format='%Y-%m-%d %H:%M:%S')
            coll.sleep_bouts['end_time'] = pd.to_datetime(coll.sleep_bouts['end_time'], format='%Y-%m-%d %H:%M:%S')

        else:
            message(f"{coll.subject_id}_{coll.coll_id}: {sleep_bouts_csv_path} does not exist",
                    level='warning', display=(not quiet), log=log, logger_name=self.log_name)
            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def activity(self, coll, quiet=False, log=True):

        message("Calculating activity levels...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        save = self.module_settings['activity']['save']
        epoch_length = self.module_settings['activity']['epoch_length']
        sedentary_gait = self.module_settings['activity']['sedentary_gait']

        coll.activity_epochs = pd.DataFrame()

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
                logger_name=self.log_name)

        cutpoint_ages = pd.DataFrame(self.module_settings['activity']['cutpoints'])

        subject_age = int(coll.subject_info['age'])
        lowpass = int(self.module_settings['activity']['lowpass'])

        cutpoint = cutpoint_ages['type'].loc[(cutpoint_ages['min_age'] <= subject_age)
                                             & (cutpoint_ages['max_age'] >= subject_age)].item()

        # get nonwear for activity_device
        device_nonwear = coll.nonwear_bouts.loc[(coll.nonwear_bouts['study_code'] == coll.study_code) &
                                                (coll.nonwear_bouts['subject_id'] == coll.subject_id) &
                                                (coll.nonwear_bouts['coll_id'] == coll.coll_id) &
                                                (coll.nonwear_bouts['device_type'] ==
                                                 coll.device_info.iloc[activity_device_index]['device_type']) &
                                                (coll.nonwear_bouts['device_location'] ==
                                                 coll.device_info.iloc[activity_device_index]['device_location']) &
                                                (coll.nonwear_bouts['event'] == 'nonwear')]

        sptw = coll.sptw
        sleep_bouts =  coll.sleep_bouts.loc[coll.sleep_bouts['bout_detect'] == 't8a4']

        e, b, avm, vm, avm_sec = activity_wrist_avm(x=coll.devices[activity_device_index].signals[accel_x_sig],
                                                    y=coll.devices[activity_device_index].signals[accel_y_sig],
                                                    z=coll.devices[activity_device_index].signals[accel_z_sig],
                                                    sample_rate=coll.devices[activity_device_index].signal_headers[accel_x_sig]['sample_rate'],
                                                    start_datetime=coll.devices[activity_device_index].header['start_datetime'],
                                                    lowpass=lowpass, epoch_length=epoch_length, cutpoint=cutpoint,
                                                    dominant=dominant, sedentary_gait=sedentary_gait,
                                                    gait=coll.gait_bouts, nonwear=device_nonwear, sptw=sptw,
                                                    sleep_bouts=sleep_bouts, quiet=quiet)

        coll.activity_epochs = e
        coll.activity_bouts = b

        # prepare avm dataframe
        coll.avm_sec = pd.DataFrame()
        coll.avm_sec['avm_num'] = np.arange(1, len(avm_sec) + 1)
        coll.avm_sec['avm'] = avm_sec
        coll.avm_sec.insert(loc=0, column='device_location',
                            value=coll.devices[activity_device_index].header['device_location'])
        coll.avm_sec = self.identify_df(coll, coll.avm_sec)

        message("Summarizing daily activity volumes...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        coll.activity_daily = activity_stats(coll.activity_bouts, quiet=quiet)

        coll.activity_epochs.insert(loc=1, column='device_location',
                                    value=coll.devices[activity_device_index].header['device_location'])
        coll.activity_epochs.insert(loc=2, column='dominant_hand', value=dominant)
        coll.activity_epochs.insert(loc=3, column='cutpoint_type', value=cutpoint)

        coll.activity_bouts.insert(loc=1, column='device_location',
                                   value=coll.devices[activity_device_index].header['device_location'])
        coll.activity_bouts.insert(loc=2, column='dominant_hand', value=dominant)
        coll.activity_bouts.insert(loc=3, column='cutpoint_type', value=cutpoint)

        coll.activity_epochs = self.identify_df(coll, coll.activity_epochs)
        coll.activity_bouts = self.identify_df(coll, coll.activity_bouts)

        coll.activity_daily.insert(loc=2, column='device_location',
                                   value=coll.devices[activity_device_index].header['device_location'])
        coll.activity_daily.insert(loc=3, column='dominant_hand', value=dominant)
        coll.activity_daily.insert(loc=4, column='cutpoint_type', value=cutpoint)
        coll.activity_daily.insert(loc=5, column='type', value='daily')

        coll.activity_daily = self.identify_df(coll, coll.activity_daily)

        # TODO: more detailed log info about what was done, epochs, days, intensities?
        # TODO: info about algortihm and settings, device used, dominant vs non-dominant, in log, methods, or data table

        if save:
            # create all file path variables
            epoch_activity_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id,
                                                          coll.coll_id, "ACTIVITY_EPOCHS"]),
                                                "csv"])
            bouts_activity_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id,
                                                          coll.coll_id, "ACTIVITY_BOUTS"]),
                                                "csv"])
            daily_activity_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id,
                                                          coll.coll_id, "ACTIVITY_DAILY"]),
                                                "csv"])
            avm_csv_name = '.'.join(['_'.join([coll.study_code, coll.subject_id,
                                                          coll.coll_id, "ACTIVITY_AVM"]),
                                                "csv"])

            epoch_activity_csv_path = self.dirs['activity_epochs'] / epoch_activity_csv_name
            bouts_activity_csv_path = self.dirs['activity_bouts'] / bouts_activity_csv_name
            daily_activity_csv_path = self.dirs['activity_daily'] / daily_activity_csv_name
            avm_csv_path = self.dirs['activity_avm'] / avm_csv_name

            epoch_activity_csv_path.parent.mkdir(parents=True, exist_ok=True)
            bouts_activity_csv_path.parent.mkdir(parents=True, exist_ok=True)
            daily_activity_csv_path.parent.mkdir(parents=True, exist_ok=True)
            avm_csv_path.parent.mkdir(parents=True, exist_ok=True)

            message(f"Saving {epoch_activity_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.activity_epochs.to_csv(epoch_activity_csv_path, index=False)

            message(f"Saving {bouts_activity_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.activity_bouts.to_csv(bouts_activity_csv_path, index=False)

            message(f"Saving {daily_activity_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.activity_daily.to_csv(daily_activity_csv_path, index=False)

            message(f"Saving {avm_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.avm_sec.to_csv(avm_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def select_activity_device(self, coll):

        # select which device to use for activity level

        dominant = self.module_settings['activity']['dominant']
        dominant_hand = coll.subject_info['dominant_hand'].lower()

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
            if dominant_hand in ['right', 'left']:

                # select dominant or non-dominant based on argument
                if dominant:
                    wrist = 'rwrist' if dominant_hand == 'right' else 'lwrist'
                else:
                    wrist = 'lwrist' if dominant_hand == 'right' else 'rwrist'

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
            if dominant_hand in ['right', 'left']:
                dominant_wrist = dominant_hand[0] + 'wrist'
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
        dominant_hand = coll.subject_info['dominant_hand'].lower()

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
            if dominant_hand in ['right', 'left']:

                # select dominant or non-dominant based on argument
                if dominant:
                    wrist = 'rwrist' if dominant_hand == 'right' else 'lwrist'
                else:
                    wrist = 'lwrist' if dominant_hand == 'right' else 'rwrist'

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
            if dominant_hand in ['right', 'left']:
                dominant_wrist = dominant_hand[0] + 'wrist'
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