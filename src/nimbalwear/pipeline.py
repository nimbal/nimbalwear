"""Run wearable data through the nimbalwear pipeline.

Classes
-------
Pipeline
    A study on which the pipeline can be run.
Collection
    A single data collection.

Exceptions
----------
NWException

Functions
---------
message(msg, level, display, log, logger_name)
    Displays or logs a message.

"""

import os
import shutil
import datetime as dt
from pathlib import Path
import logging
import traceback
from functools import wraps
import operator
import subprocess
from shutil import copyfile

import toml
from tqdm import tqdm
import numpy as np
import pandas as pd
from isodate import parse_duration

from .data import Device
from .nonwear import detach_nonwear, nonwear_stats
from .sleep import detect_sptw, detect_sleep_bouts, sptw_stats
from .gait import detect_steps, define_bouts, gait_stats
from .activity import activity_wrist_avm, activity_stats
from .utils import update_dict
from .reports import collection_report as cr

from .__version__ import __version__

COLLECTION_COLS = ['study_code', 'subject_id', 'coll_id', 'dominant_hand', 'age', 'var_1', 'var_2', 'var_3']
DEVICE_COLS = ['study_code', 'subject_id', 'coll_id', 'device_type', 'device_id', 'device_location', 'file_name']

class Study:
    """"Represents a study on which the pipeline can be run.

    Attributes
    ----------
    quiet : bool
        Suppress displayed messages.
    log : bool
        Log messages.
    study_dir : Path or str
        Directory where study is stored.
    study_code : str
        Unique study identifier.
    settings_path : Path or str
        Path to the file containing custom settings.
    default_study_settings_path : Path or str
        Patht to the file contatining default settings for this study.
    dirs : dict
        Dictionary of directories within the study_dir used by the pipeline to store data.
    device_info_path : Path or str
        Path to file containing information about each device in each collection in the study.
    collection_info_path : Path or str
        Path to file containing information about each collection in the study.
    status_path : Path or str
        Path to file containing information about the pipeline status of each collection.
    study_settings_str : str
        String version of settings toml file, for use in logs.
    device_info : DataFrame
        Information about each device in each collection in the study.
    collection_info : DataFrame
        Information about each collection in the study.


    """

    def __init__(self, study_dir, settings_path=None, create=False):
        """Read settings, devices, and collections file to construct Pipeline instance.

        Parameters
        ----------
        study_dir : Path or str
            Directory where study is stored.
        settings_path : Path or str, optional
            Path to the file containing the default settings for the study, defaults to None in which default settings file
            path relative to study_dir is used.

        """

        self.quiet = False
        self.log = True

        self.study_dir = study_dir = Path(study_dir)
        self.study_code = study_dir.stem
        self.settings_path = Path(settings_path) if settings_path is not None else settings_path

        if create:
            self._create_study()

        self._load_study()

        return

    def _create_study(self):

        study_dir = self.study_dir

        isdirexists = study_dir.is_dir()

        if isdirexists:
            print(f"{study_dir} cannot be created because it already exists.")
            return

        print(f"Creating {study_dir}...")
        study_dir.mkdir(parents=True, exist_ok=True)
        dotnimbalwear_path = study_dir / ".nimbalwear"
        open(dotnimbalwear_path, 'a').close()  # creates empty file
        if os.name == "nt":
            subprocess.run(["attrib", "+H", dotnimbalwear_path], check=True)  # hides file if os is windows

        return

    def _load_study(self):

        study_dir = self.study_dir

        isdirstudy = (study_dir / ".nimbalwear").is_file()

        if not isdirstudy:
            raise FileNotFoundError(f"{study_dir} cannot be loaded because it does not contain a nimbalwear study.")

        print(f"Loading study from {study_dir}...")

        self._load_settings()

        self._update_dirs()

        self._init_structure()

        return

    def _load_settings(self):

        study_dir= self.study_dir
        study_code = self.study_code
        settings_path = self.settings_path

        # LOAD DEFAULT SETTINGS
        default_settings_path = Path(__file__).parent.absolute() / 'settings/settings.toml'
        print("Loading nimbalwear default settings...")
        with open(default_settings_path, 'r') as f:
            default_settings = toml.load(f)
        self.study_settings = default_settings.copy()

        # LOAD STUDY SETTINGS
        self.default_study_settings_path = default_study_settings_path = study_dir / 'study/settings/settings.toml'
        if not default_study_settings_path.is_file():
            print(f"No {study_code} default settings exist. " +
                  "Copying nimbalwear default settings to {default_study_settings_path}...")
            default_study_settings_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(default_settings_path, default_study_settings_path)
        else:
            print(f"Updating {study_code} default settings...")

        with open(default_study_settings_path, 'r') as f:
            default_study_settings = toml.load(f)

        self.default_study_settings = default_study_settings
        self._update_settings(default_study_settings)

        # LOAD CUSTOM SETTINGS

        # set settings_path to None if it does not exist
        if (settings_path is not None) and (not settings_path.is_file()):
            print(f"Cannot update custom settings: {settings_path} does not exist.")
            settings_path = None

        if settings_path is not None:
            print("Updating custom settings...")
            with open(settings_path, 'r') as f:
                custom_settings = toml.load(f)
            self._update_settings(custom_settings)

        return

    def _update_settings(self, new_settings):

        self.study_settings = update_dict(self.study_settings, new_settings)
        self.study_settings_str = toml.dumps(self.study_settings)

        return

    def _update_dirs(self):

        study_dir = self.study_dir
        study_settings = self.study_settings

        # parse specific settings into Pipeline attributes
        dirs = study_settings['study']['dirs']
        self.dirs = {key: study_dir / value for key, value in dirs.items()}

        abs_dirs = study_settings['study'].get('abs_dirs', {})
        self.dirs.update({key: Path(value) for key, value in abs_dirs.items()})

        return

    def _init_structure(self):

        dirs = self.dirs

        # initialize folder structure
        for key, value in dirs.items():
            Path(value).mkdir(parents=True, exist_ok=True)

        # OPEN COLLECTION AND DEVICE FILES

        # pipeline data file paths
        self.device_info_path = device_info_path = dirs['study'] / 'devices.csv'
        self.collection_info_path = collection_info_path = dirs['study'] / 'collections.csv'
        self.status_path = dirs['study'] / 'status.csv'

        # read data files if they exist or create blanks
        if collection_info_path.is_file():
            print("Reading collection info...")
            collection_info = pd.read_csv(collection_info_path, dtype=str).fillna('')
        else:
            print(f"No collection info exists. Creating {collection_info_path}")
            collection_info = pd.DataFrame(columns=COLLECTION_COLS)
            collection_info.to_csv(collection_info_path, index=False)
        self.collection_info = collection_info

        if device_info_path.is_file():
            print("Reading device info...")
            device_info = pd.read_csv(device_info_path, dtype=str).fillna('')
        else:
            print(f"No device info exists. Creating {device_info_path}")
            device_info = pd.DataFrame(columns=DEVICE_COLS)
            device_info.to_csv(device_info_path, index=False)
        self.device_info = device_info

        return

    def sync_raw(self, raw_source_dir=None, update_source=False, update_default=False):

        dirs = self.dirs

        if raw_source_dir is None:
            raw_source_dir = dirs.get('raw_source', None)
        else:
            if update_default:
                self._update_default_raw_source(raw_source_dir)
            if update_source:
                self._update_raw_source(raw_source_dir)

            raw_source_dir = Path(raw_source_dir)

        if raw_source_dir is None:
            print("Could not sync raw data: No known source.")
            return
        elif not raw_source_dir.is_dir():
            print(f"Could not sync raw data: {raw_source_dir} does not exist.")
            return

        source_files = [f.name for f in raw_source_dir.iterdir() if f.is_file() and not f.stem.startswith('.')]

        dest_dir = dirs['device_raw']
        dest_files = [f.name for f in dest_dir.iterdir()]

        source_files = [f for f in source_files if f not in dest_files]

        print(f"Syncing raw data: Copying {len(source_files)} filesfrom {raw_source_dir} to {dest_dir}...")

        for f in source_files:
            src = raw_source_dir / f
            dst = dest_dir / f

            print(f"Copying {f}...")

            copyfile(src, dst)

        return

    def _update_default_raw_source(self, raw_source_dir):

        # set and write raw_source to default_study_settings
        abs_dirs = self.default_study_settings['study'].get('abs_dirs', {})
        abs_dirs['raw_source'] = raw_source_dir
        self.default_study_settings['study']['abs_dirs'] = abs_dirs
        self._write_study_settings()
        print(f"Default raw data source set to {raw_source_dir}.")

        return

    def _update_raw_source(self, raw_source_dir):

        # set and write raw_source to default_study_settings

        new_settings = {'study': {'abs_dirs': {'raw_source': raw_source_dir}}}
        self._update_settings(new_settings)
        self._update_dirs()
        print(f"Raw data source set to {raw_source_dir}.")

        return

    def _write_study_settings(self):

        with open(self.default_study_settings_path, 'w') as f:
            toml.dump(self.default_study_settings, f)

        return

    def coll_status(f):
        @wraps(f)
        def coll_status_wrapper(self, *args, **kwargs):

            # the keys are the same as the function names
            coll_status = {
                'subject_id': kwargs['coll'].subject_id,
                'coll_id': kwargs['coll'].coll_id,
                'convert': '',
                'prep': '',
                'analytics': '',
            }

            status_df = pd.read_csv(self.status_path, dtype=str) if self.status_path.exists() \
                else pd.DataFrame(columns=coll_status.keys())

            current_collection = (coll_status['subject_id'], coll_status['coll_id'])
            collections = zip(status_df['subject_id'], status_df['coll_id'])

            if current_collection in collections:
                index = status_df.loc[(status_df['subject_id'] == coll_status['subject_id'])
                                      & (status_df['coll_id'] == coll_status['coll_id'])].index[0]
                coll_status = status_df.to_dict(orient='records')[index]
            else:
                index = (status_df.index.max() + 1)

            try:
                res = f(self, *args, **kwargs)
                coll_status[f.__name__] = 'Success'
                return res
            except NWException as e:
                coll_status[f.__name__] = f'Failed'
                message(str(e), level='warning', display=(not kwargs['quiet']), log=kwargs['log'], logger_name=self.log_name)
                message('', level='info', display=(not kwargs['quiet']), log=kwargs['log'], logger_name=self.log_name)
                return kwargs['coll']
            except Exception as e:
                coll_status[f.__name__] = f'Failed'
                raise e
            finally:
                status_df.loc[index, list(coll_status.keys())] = list(coll_status.values())
                status_df.to_csv(self.status_path, index=False)

        return coll_status_wrapper

    def run_pipeline(self, collections=None, stages=None, settings_path=None, supp_pwd= None, quiet=False, log=True,
                     log_level=logging.INFO):
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

            # set up logger
            start_time = dt.datetime.now()
            self.log_name = f'{subject_id}_{coll_id}_{start_time.strftime("%Y%m%d%H%M%S")}'
            log_path = self.dirs['logs'] / (self.log_name + '.log')
            settings_dump_path = self.dirs['logs'] / (self.log_name + '_settings.txt')

            fileh = logging.FileHandler(log_path, 'a')
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            fileh.setFormatter(formatter)
            fileh.setLevel(log_level)

            logger = logging.getLogger(self.log_name)
            for hdlr in logger.handlers[:]:  # remove all old handlers
                logger.removeHandler(hdlr)
            logger.setLevel(log_level)
            logger.addHandler(fileh)

            # display header messages
            message("\n\n", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message(f"---- Processing collection ----------------------------------------------",
                    level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message(f"---- Study {self.study_code}, Subject {subject_id}, Collection {coll_id} --------", level='info', display=(not self.quiet),
                    log=self.log, logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)

            message(f"nimbalwear v{__version__}", level='info', display=(not self.quiet), log=self.log,
                    logger_name=self.log_name)

            if not isinstance(self.collection_info, pd.DataFrame):
                message("Missing collection info file in meta folder `collections.csv`", level='warning',
                        display=(not self.quiet), log=self.log, logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)


            # set/reset default study settings for pipeline
            self.pipeline_settings = self.study_settings.copy()

            # update custom pipeline settings if file or 'auto' passed
            if settings_path not in (None, 'auto'):

                # convert settings_path to Path
                settings_path = Path(settings_path)

                # if settings file exists then update settings else display message
                if settings_path.is_file():

                    with open(settings_path, 'r') as f:
                        pipeline_settings_dict = toml.load(f)
                    self.pipeline_settings = update_dict(self.pipeline_settings, pipeline_settings_dict)

                else:
                    message(f"Custom settings file {settings_path} does not exist.", level='warning',
                            display=(not self.quiet), log=self.log, logger_name=self.log_name)
                    message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)

            elif settings_path == 'auto':
                # look for custom settings file based on subject_id and coll_id
                study_settings_dir = self.default_study_settings_path.parent
                study_settings_name = self.default_study_settings_path.name
                coll_settings_path = study_settings_dir / f"{self.study_code}_{subject_id}_{coll_id}_{study_settings_name}"

                # if file exists update settings
                if coll_settings_path.is_file():

                    with open(coll_settings_path, 'r') as f:
                        coll_settings_dict = toml.load(f)
                    self.pipeline_settings = update_dict(self.pipeline_settings, coll_settings_dict)

            if stages is None:
                stages = self.pipeline_settings['pipeline']['stages']
            else:
                self.pipeline_settings['pipeline']['stages'] = stages

            self.pipeline_settings_str = toml.dumps(self.pipeline_settings)

            message(f"Stages: {stages}", level='info', display=(not self.quiet), log=self.log,
                    logger_name=self.log_name)
            message("", level='info', display=(not self.quiet), log=self.log, logger_name=self.log_name)

            #TODO: check for valid stages and order - up to user now

            # dump settings to file

            with open(settings_dump_path, "w") as f:
                f.write(f"Study {self.study_code}, Subject {subject_id}, Collection {coll_id}, Time {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.pipeline_settings_str)

            try:
                # get devices for this collection from device_list
                coll_device_list_df = self.device_info.loc[(self.device_info['study_code'] == self.study_code) &
                                                           (self.device_info['subject_id'] == subject_id) &
                                                           (self.device_info['coll_id'] == coll_id)]
                coll_device_list_df.reset_index(inplace=True, drop=True)

                coll_subject_dict = {}
                if isinstance(self.collection_info, pd.DataFrame):
                    coll_subject_df = self.collection_info.loc[(self.collection_info['study_code'] == self.study_code) &
                                                               (self.collection_info['subject_id'] == subject_id) &
                                                               (self.collection_info['coll_id'] == coll_id)]
                    coll_subject_df.reset_index(inplace=True, drop=True)
                    coll_subject_dict = coll_subject_df.iloc[0].to_dict() if coll_subject_df.shape[0] > 0 else {}

                # construct collection class and process
                coll = Collection(study_code=self.study_code, subject_id=subject_id, coll_id=coll_id)

                coll.device_info = coll_device_list_df
                coll.collection_info = coll_subject_dict
                coll.supp_pwd = supp_pwd

                self.process_collection(coll=coll, stages=stages)

            except:
                tb = traceback.format_exc()
                message(tb, level='error', display=(not self.quiet), log=self.log, logger_name=self.log_name)

            del coll

            message("---- End ----------------------------------------------\n", level='info', display=(not self.quiet),
                    log=self.log, logger_name=self.log_name)

        return

    def process_collection(self, coll, stages):
        """Processes the collection

        Args:
            coll:
            single_stage (str): None, 'read', 'nonwear', 'crop', 'save_sensors', 'activity', 'gait', 'sleep, 'posture'
            ...
        Returns:
            True if successful, False otherwise.
        """

        stage_switch = {'convert': lambda: self.convert(coll=coll, quiet=self.quiet, log=self.log),
                        'prep': lambda: self.prep(coll=coll, quiet=self.quiet, log=self.log),
                        'analytics': lambda: self.analytics(coll=coll, quiet=self.quiet, log=self.log),
                        'reports': lambda: self.reports(coll=coll, quiet=self.quiet, log=self.log),}

        # if single_stage in ['activity', 'gait', 'sleep']:
        #     coll = self.required_devices(coll=coll, single_stage=single_stage, quiet=self.quiet, log=self.log)

        read_stages = ['convert', 'prep', 'analytics']
        if any([rs in stages for rs in read_stages]):
            # read data from all devices in collection
            coll = self.read(coll=coll, stages=stages, quiet=self.quiet, log=self.log)

        for stage in stages:
            coll = stage_switch.get(stage, lambda: 'Invalid')()

        return True

    def read(self, coll, stages, quiet=False, log=True):

        message("---- Reading device data --------", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        overwrite_header = self.pipeline_settings['modules']['read']['overwrite_header']

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

            if stages[0] == 'convert':

                device_file_path = self.dirs['device_raw'] / device_file_name
                import_func = import_switch.get(device_type, lambda: 'Invalid')

                #TODO: Rotate GENEActiv 90 deg if location is ankle

            elif stages[0] == 'prep':

                device_file_path = self.dirs['device_edf_raw'] / device_edf_name
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

        if stages[0] == 'analytics':
            coll = self.read_nonwear(coll=coll, quiet=self.quiet, log=self.log)

        return coll

    @coll_status
    def convert(self, coll, quiet=False, log=True):

        message("---- Convert stage --------", level='info', display=(not self.quiet), log=self.log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        coll = self.save_devices(coll=coll, dir=self.dirs['device_edf_raw'], quiet=self.quiet, log=self.log)

        return coll

    @coll_status
    def prep(self, coll, quiet=False, log=True):

        message("---- Data preparation stage --------", level='info', display=(not self.quiet), log=self.log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        if self.pipeline_settings['modules']['prep']['adj_start']:
            coll = self.adj_start(coll, quiet=quiet, log=log)

        if self.pipeline_settings['modules']['prep']['autocal']:
            coll = self.autocal(coll, quiet=quiet, log=log)

        if self.pipeline_settings['modules']['prep']['sync']:
            coll = self.sync(coll, quiet=quiet, log=log)



        coll = self.save_devices(coll=coll, dir=self.dirs['device_edf_standard'], quiet=self.quiet, log=self.log)

        # message("Saving standardized device data to EDF...", level='info', display=(not quiet), log=log,
        #         logger_name=self.log_name)
        # message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
        #
        # for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
        #                        desc='Saving standardized device data to EDF'):
        #
        #     study_code = row['study_code']
        #     subject_id = row['subject_id']
        #     coll_id = row['coll_id']
        #     device_type = row['device_type']
        #     device_location = row['device_location']
        #     device_edf_name = f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}.edf"
        #
        #     # create all file path variables
        #     standard_device_path = self.dirs['device_edf_standard'] / device_edf_name
        #
        #     # check that all folders exist for data output files
        #     standard_device_path.parent.mkdir(parents=True, exist_ok=True)
        #
        #     message(f"Saving {standard_device_path}", level='info', display=(not quiet), log=log,
        #             logger_name=self.log_name)
        #
        #     # write device data as edf
        #     coll.devices[index].export_edf(file_path=standard_device_path, quiet=quiet)
        #
        #     message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        if self.pipeline_settings['modules']['prep']['nonwear']:
            coll = self.nonwear(coll=coll, quiet=quiet, log=log)

        if self.pipeline_settings['modules']['prep']['crop']:
            coll = self.crop(coll=coll, quiet=quiet, log=log)

        coll = self.save_devices(coll=coll, dir=self.dirs['device_edf_cropped'], quiet=self.quiet, log=self.log)
        #
        # message("Saving cropped device data to EDF...", level='info', display=(not quiet), log=log,
        #         logger_name=self.log_name)
        # message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
        #
        # for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
        #                        desc='Saving cropped device data to EDF'):
        #     study_code = row['study_code']
        #     subject_id = row['subject_id']
        #     coll_id = row['coll_id']
        #     device_type = row['device_type']
        #     device_location = row['device_location']
        #     device_edf_name = f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}.edf"
        #
        #     # create all file path variables
        #     cropped_device_path = self.dirs['device_edf_cropped'] / device_edf_name
        #
        #     # check that all folders exist for data output files
        #     cropped_device_path.parent.mkdir(parents=True, exist_ok=True)
        #
        #     message(f"Saving {cropped_device_path}", level='info', display=(not quiet), log=log,
        #             logger_name=self.log_name)
        #
        #     # write device data as edf
        #     coll.devices[index].export_edf(file_path=cropped_device_path, quiet=quiet)
        #
        #     message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        if self.pipeline_settings['modules']['prep']['save_sensors']:
            coll = self.save_sensors(coll=coll, dir=self.dirs['sensor_edf'], quiet=self.quiet, log=self.log)

        return coll

    @coll_status
    def analytics(self, coll, quiet=False, log=True):

        message("---- Analytics stage --------", level='info', display=(not self.quiet), log=self.log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        # process gait
        if self.pipeline_settings['modules']['analytics']['gait']:
            coll = self.gait(coll=coll, quiet=self.quiet, log=self.log, )

        # process sleep
        if self.pipeline_settings['modules']['analytics']['sleep']:
            coll = self.sleep(coll=coll, quiet=self.quiet, log=self.log)

        # process activity levels
        if self.pipeline_settings['modules']['analytics']['activity']:
            coll = self.activity(coll=coll, quiet=self.quiet, log=self.log)

        return coll

    def reports(self, coll, quiet=False, log=True):

        message("---- Reports stage --------", level='info', display=(not self.quiet), log=self.log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        # create collection report
        if self.pipeline_settings['modules']['reports']['collection_report']:
            coll = self.collection_report(coll=coll, quiet=self.quiet, log=self.log, )

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

            pre_err, post_err, iter, offset, scale, tempoffset = coll.devices[idx].autocal(quiet=quiet)

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
                                  'pre_err': pre_err, 'post_err': post_err, 'iter': iter,
                                  'offset_x': offset[0], 'offset_y': offset[1], 'offset_z': offset[2],
                                  'scale_x': scale[0], 'scale_y': scale[1], 'scale_z': scale[2],
                                  'tempoffset_x': tempoffset[0], 'tempoffset_y': tempoffset[1], 'tempoffset_z': tempoffset[2],
                                  }, index=[0])

            message(f"Autocalibrated {device_type} {device_location}: Calibration error reduced from {pre_err} to {post_err} after {iter} iterations.",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            if self.pipeline_settings['modules']['autocal']['save']:

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

        sync_type = self.pipeline_settings['modules']['sync']['type']
        sync_at_config = self.pipeline_settings['modules']['sync']['sync_at_config']
        search_radius = self.pipeline_settings['modules']['sync'].get('search_radius', None)
        rest_min = self.pipeline_settings['modules']['sync']['rest_min']
        rest_max = self.pipeline_settings['modules']['sync']['rest_max']
        rest_sens = self.pipeline_settings['modules']['sync']['rest_sens']
        flip_max = self.pipeline_settings['modules']['sync']['flip_max']
        min_flips = self.pipeline_settings['modules']['sync']['min_flips']
        reject_above_ae = self.pipeline_settings['modules']['sync']['reject_above_ae']
        req_tgt_corr = self.pipeline_settings['modules']['sync']['req_tgt_corr']

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

                accel_idx = coll.devices[idx].get_signal_index(self.pipeline_settings['pipeline']['sensors']['accelerometer']['signals'][0])

                # set signal_ds to downsample to somewhere between 5-11 Hz for sync detection if possible
                freq = coll.devices[idx].signal_headers[accel_idx]['sample_rate']
                try:
                    ds_index = [freq % x for x in range(5, 12)].index(0)
                except ValueError:
                    ds_index = freq - 5
                signal_ds = round(freq / (5 + ds_index))

                # get signal labels and add 'Config' to end for sig_idx -1
                accel_signal_labels = tuple(self.pipeline_settings['pipeline']['sensors']['accelerometer']['signals']) + ('Config', )

                syncs, segments = coll.devices[idx].sync(ref=coll.devices[0],
                                                         sig_labels=tuple(self.pipeline_settings['pipeline']['sensors']['accelerometer']['signals']),
                                                         sync_type=sync_type, sync_at_config=sync_at_config,
                                                         search_radius=search_radius, signal_ds=signal_ds,
                                                         rest_min=rest_min, rest_max=rest_max, rest_sens=rest_sens,
                                                         flip_max=flip_max, min_flips=min_flips,
                                                         reject_above_ae=reject_above_ae, req_tgt_corr=req_tgt_corr)


                message(f"Synchronized {device_type} {device_location} to {ref_device_type} {ref_device_location} at {syncs.shape[0]} sync points",
                        level='info', display=(not quiet), log=log, logger_name=self.log_name)

                ref_start_datetime = coll.devices[0].header['start_datetime']
                sync_start_time = []
                sync_end_time = []

                for i, r in syncs.iterrows():

                    ref_sig_idx = 0 if r['ref_sig_idx'] < 0 else int(r['ref_sig_idx'])

                    sig_idx = coll.devices[0].get_signal_index(accel_signal_labels[ref_sig_idx])
                    sample_rate = coll.devices[0].signal_headers[sig_idx]['sample_rate']

                    sync_start_time.append(ref_start_datetime + dt.timedelta(seconds=(r['ref_start_idx'] / sample_rate)))
                    sync_end_time.append(ref_start_datetime + dt.timedelta(seconds=(r['ref_end_idx'] / sample_rate)))

                sync_ref_sig_labels = [accel_signal_labels[int(r['ref_sig_idx'])] for i, r in syncs.iterrows()]
                sync_tgt_sig_labels = [accel_signal_labels[int(r['tgt_sig_idx'])] for i, r in syncs.iterrows()]

                syncs.insert(loc=0, column='study_code', value=study_code)
                syncs.insert(loc=1, column='subject_id', value=subject_id)
                syncs.insert(loc=2, column='coll_id', value=coll_id)
                syncs.insert(loc=3, column='device_type', value=device_type)
                syncs.insert(loc=4, column='device_location', value=device_location)
                syncs.insert(loc=5, column='sync_id', value=range(1, syncs.shape[0] + 1))
                syncs.insert(loc=6, column='start_time', value=sync_start_time)
                syncs.insert(loc=7, column='end_time', value=sync_end_time)
                syncs.insert(loc=8, column='ref_device_type', value=ref_device_type)
                syncs.insert(loc=9, column='ref_device_location', value=ref_device_location)
                syncs.insert(loc=11, column='ref_sig_label', value=sync_ref_sig_labels)
                syncs.insert(loc=17, column='tgt_sig_label', value=sync_tgt_sig_labels)

                seg_start_time = []
                seg_end_time = []

                for i, r in segments.iterrows():
                    sig_idx = coll.devices[0].get_signal_index(accel_signal_labels[0])
                    sample_rate = coll.devices[0].signal_headers[sig_idx]['sample_rate']

                    seg_start_time.append(ref_start_datetime + dt.timedelta(seconds=(r['ref_start_idx'] / sample_rate)))
                    seg_end_time.append(ref_start_datetime + dt.timedelta(seconds=(r['ref_end_idx'] / sample_rate)))

                segments.insert(loc=0, column='study_code', value=study_code)
                segments.insert(loc=1, column='subject_id', value=subject_id)
                segments.insert(loc=2, column='coll_id', value=coll_id)
                segments.insert(loc=3, column='device_type', value=device_type)
                segments.insert(loc=4, column='device_location', value=device_location)
                segments.insert(loc=5, column='segment_id', value=range(1, segments.shape[0] + 1))
                segments.insert(loc=6, column='start_time', value=seg_start_time)
                segments.insert(loc=7, column='end_time', value=seg_end_time)

                if self.pipeline_settings['modules']['sync']['save']:

                    # create all file path variables
                    syncs_csv_name = (f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}"
                                      + "_SYNC_EVENTS.csv")

                    segments_csv_name = (f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}"
                                         + "_SYNC_SEGMENTS.csv")

                    syncs_csv_path = self.dirs['sync_events'] / syncs_csv_name
                    segments_csv_path = self.dirs['sync_segments'] / segments_csv_name

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
        duration_iso = self.pipeline_settings['modules']['prep']['adj_start']

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
        """Detect wear and non-wear bouts for all devices in the collection.

        Parameters
        ----------
        coll : Collection
            Collection object containing attributes and methods related to the collection
        quiet : bool, optional
            Suppress displayed messages (default is False)
        log : bool, optional
            Log messages (default is True)

        """

        # process nonwear for all devices
        message("Detecting non-wear...", level='info', display=(not quiet), log=log, logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        save = self.pipeline_settings['modules']['nonwear']['save']

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

            # find device body location type

            # get location aliases from settings
            wrist_locations = (self.pipeline_settings['pipeline']['device_locations']['rwrist']['aliases']
                               + self.pipeline_settings['pipeline']['device_locations']['lwrist']['aliases'])
            ankle_locations = (self.pipeline_settings['pipeline']['device_locations']['rankle']['aliases']
                               + self.pipeline_settings['pipeline']['device_locations']['lankle']['aliases'])
            chest_locations = self.pipeline_settings['pipeline']['device_locations']['chest']['aliases']

            # compare device location to location types
            if device_location.upper() in wrist_locations:
                location_type = "wrist"
            elif device_location.upper() in ankle_locations:
                location_type = "ankle"
            elif device_location.upper() in chest_locations:
                location_type = "chest"

            # get location specific non-wear settings
            accel_std_thresh_mg = self.pipeline_settings['modules']['nonwear']['settings'][location_type]['accel_std_thresh_mg']
            low_temperature_cutoff = self.pipeline_settings['modules']['nonwear']['settings'][location_type]['low_temperature_cutoff']
            high_temperature_cutoff = self.pipeline_settings['modules']['nonwear']['settings'][location_type]['high_temperature_cutoff']
            temp_dec_roc = self.pipeline_settings['modules']['nonwear']['settings'][location_type]['temp_dec_roc']
            temp_inc_roc = self.pipeline_settings['modules']['nonwear']['settings'][location_type]['temp_inc_roc']

            # current device
            device = coll.devices[i]

            # check for data loaded
            if device is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            # get signal indices
            accel_x_idx = device.get_signal_index('Accelerometer x')
            accel_y_idx = device.get_signal_index('Accelerometer y')
            accel_z_idx = device.get_signal_index('Accelerometer z')
            temperature_idx = device.get_signal_index('Temperature')

            # check for all required signals
            if None in [accel_x_idx, accel_y_idx, accel_z_idx, temperature_idx]:
                message(f"{device_type}_{device_location} does not contain all signals required to edetect non-wear",
                        level='info', display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            # get signals
            accel_x = device.signals[accel_x_idx]
            accel_y = device.signals[accel_y_idx]
            accel_z = device.signals[accel_z_idx]
            temperature = device.signals[temperature_idx]

            # TODO: index signals by label - make option to return datetimeindex

            # get sample rates
            accel_fs = device.signal_headers[accel_x_idx]['sample_rate']
            temperature_fs = device.signal_headers[temperature_idx]['sample_rate']

            # detect non-wear using DETACH algorithm
            nonwear_bouts, nonwear_array = detach_nonwear(x_values=accel_x, y_values=accel_y, z_values=accel_z,
                                                          temperature_values=temperature, accel_freq=accel_fs,
                                                          temperature_freq=temperature_fs,
                                                          std_thresh_mg=accel_std_thresh_mg,
                                                          low_temperature_cutoff=low_temperature_cutoff,
                                                          high_temperature_cutoff=high_temperature_cutoff,
                                                          temp_dec_roc=temp_dec_roc, temp_inc_roc=temp_inc_roc,
                                                          quiet=quiet)
            algorithm_name = 'DETACH'

            # label non-wear bouts as non-wear events
            nonwear_bouts['event'] = "nonwear"

            nonwear_bouts.rename(columns={'Start Datapoint': 'start_datapoint', 'End Datapoint': 'end_datapoint'},
                                 inplace=True)

            # count bouts
            bout_count = nonwear_bouts.shape[0]

            message(f"Detected {bout_count} nonwear bouts for {device_type} {device_location} ({algorithm_name})",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            # convert datapoints to times and insert into dataframe as start and end time for each event
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

            # select columns
            nonwear_bouts = nonwear_bouts[['event', 'start_time', 'end_time']]

            # calculate wear events and insert between non-wear events

            # nonwear end times are wear start times -- nonwear start times are wear end times
            wear_start_times = nonwear_end_times
            wear_end_times = nonwear_start_times

            # collection start is first wear start
            wear_start_times.insert(0, start_date)

            # collection end is last wear end
            wear_end_times.append(end_date)

            # remove first and last wear bout if duration is 0 - started or ended with non-wear with non-wear
            if wear_start_times[0] == wear_end_times[0]:
                wear_start_times = wear_start_times[1:]
                wear_end_times = wear_end_times[1:]

            if wear_start_times[-1] == wear_end_times[-1]:
                wear_start_times = wear_start_times[:-1]
                wear_end_times = wear_end_times[:-1]

            # create wear dataframe
            wear_bouts = pd.DataFrame({'start_time': wear_start_times, 'end_time': wear_end_times, })
            wear_bouts['event'] = 'wear'

            # concatenate with nonwear and sort by start_time
            nonwear_bouts = pd.concat([nonwear_bouts, wear_bouts], ignore_index=True)
            nonwear_bouts = nonwear_bouts.sort_values('start_time')

            # number bouts as id
            nonwear_bouts.insert(loc=0, column='id', value=range(1, nonwear_bouts.shape[0] + 1))

            # calculate daily summary non-wear
            daily_nonwear = nonwear_stats(nonwear_bouts, quiet=quiet)

            # add identifiers and settings to bouts and daily summary
            nonwear_bouts.insert(loc=0, column='study_code', value=study_code)
            nonwear_bouts.insert(loc=1, column='subject_id', value=subject_id)
            nonwear_bouts.insert(loc=2, column='coll_id', value=coll_id)
            nonwear_bouts.insert(loc=3, column='device_type', value=device_type)
            nonwear_bouts.insert(loc=4, column='device_location', value=device_location)
            nonwear_bouts.insert(loc=5, column='accel_std_thresh_mg', value=accel_std_thresh_mg)
            nonwear_bouts.insert(loc=6, column='low_temperature_cutoff', value=low_temperature_cutoff)
            nonwear_bouts.insert(loc=7, column='high_temperature_cutoff', value=high_temperature_cutoff)
            nonwear_bouts.insert(loc=8, column='temp_dec_roc', value=temp_dec_roc)
            nonwear_bouts.insert(loc=9, column='temp_inc_roc', value=temp_inc_roc)

            daily_nonwear.insert(loc=0, column='study_code', value=study_code)
            daily_nonwear.insert(loc=1, column='subject_id', value=subject_id)
            daily_nonwear.insert(loc=2, column='coll_id', value=coll_id)
            daily_nonwear.insert(loc=3, column='device_type', value=device_type)
            daily_nonwear.insert(loc=4, column='device_location', value=device_location)
            daily_nonwear.insert(loc=5, column='accel_std_thresh_mg', value=accel_std_thresh_mg)
            daily_nonwear.insert(loc=6, column='low_temperature_cutoff', value=low_temperature_cutoff)
            daily_nonwear.insert(loc=7, column='high_temperature_cutoff', value=high_temperature_cutoff)
            daily_nonwear.insert(loc=8, column='temp_dec_roc', value=temp_dec_roc)
            daily_nonwear.insert(loc=9, column='temp_inc_roc', value=temp_inc_roc)

            # append to collection attribute
            coll.nonwear_bouts = pd.concat([coll.nonwear_bouts, nonwear_bouts], ignore_index=True)
            coll.daily_nonwear = pd.concat([coll.daily_nonwear, daily_nonwear], ignore_index=True)

            # save output files
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

                # create parent folders if they don't exist
                nonwear_csv_path.parent.mkdir(parents=True, exist_ok=True)
                nonwear_daily_csv_path.parent.mkdir(parents=True, exist_ok=True)

                # save files
                message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)
                nonwear_bouts.to_csv(nonwear_csv_path, index=False)

                message(f"Saving {nonwear_daily_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)
                daily_nonwear.to_csv(nonwear_daily_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def read_nonwear(self, coll, quiet=False, log=True):

        # read nonwear data for all devices
        message("Reading non-wear data from files...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        # if single_stage == 'crop':
        #     nonwear_csv_dir = self.dirs['nonwear_bouts_standard']
        # else:
        #     nonwear_csv_dir = self.dirs['nonwear_bouts_cropped']

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
            nonwear_bouts['start_time'] = pd.to_datetime(nonwear_bouts['start_time'], yearfirst=True)
            nonwear_bouts['end_time'] = pd.to_datetime(nonwear_bouts['end_time'], yearfirst=True)

            # append to collection attribute
            coll.nonwear_bouts = pd.concat([coll.nonwear_bouts, nonwear_bouts], ignore_index=True)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def crop(self, coll, quiet=False, log=True):
        """Crop non-wear from start and end of all devices in the collection.

        Parameters
        ----------
        coll : Collection
            Collection object containing attributes and methods related to the collection
        quiet : bool, optional
            Suppress displayed messages (default is False)
        log : bool, optional
            Log messages (default is True)

        """

        message("Cropping initial and final non-wear...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)


        # get crop settings
        min_wear_time = self.pipeline_settings['modules']['crop']['min_wear_time']
        save = self.pipeline_settings['modules']['crop']['save']

        # make copy of nonwear bouts dataframe
        nonwear_bouts = coll.nonwear_bouts.copy()

        # if there is nonwear data for any devices in this collection
        if not nonwear_bouts.empty:
            nonwear_bouts['duration'] = ((nonwear_bouts['end_time'] - nonwear_bouts['start_time']).dt.total_seconds() / 60).round()

        # re-initialize collection daily_nonwear and nonwear_bouts dataframes
        coll.daily_nonwear = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'device_type',
                                                   'device_location', 'accel_std_thresh_mg', 'low_temperature_cutoff',
                                                   'high_temperature_cutoff', 'temp_dec_roc', 'temp_inc_roc',
                                                   'day_num', 'date', 'wear', 'nonwear'])

        coll.nonwear_bouts = pd.DataFrame(columns=['study_code', 'subject_id', 'coll_id', 'device_type',
                                                   'device_location', 'accel_std_thresh_mg', 'low_temperature_cutoff',
                                                   'high_temperature_cutoff', 'temp_dec_roc', 'temp_inc_roc', 'id',
                                                   'event', 'start_time', 'end_time'])

        # loop through all devices in collection
        for i, r in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Cropping initial and final non-wear'):

            # get device info from device list
            study_code = r['study_code']
            subject_id = r['subject_id']
            coll_id = r['coll_id']
            device_type = r['device_type']
            device_location = r['device_location']

            # get device data object
            device = coll.devices[i]

            # if device data object doesn't exist display warning and go to next device
            if device is None:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No device data",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            # if there is nonwear data for any devices in this collection
            if not nonwear_bouts.empty:

                # initialize daily nonwear dataframe
                daily_nonwear = pd.DataFrame(columns=['day_num', 'date', 'wear', 'nonwear'])

                # get nonwear bouts for current device
                device_bouts = nonwear_bouts.loc[(nonwear_bouts['study_code'] == study_code) &
                                                 (nonwear_bouts['subject_id'] == subject_id) &
                                                 (nonwear_bouts['coll_id'] == coll_id) &
                                                 (nonwear_bouts['device_type'] == device_type) &
                                                 (nonwear_bouts['device_location'] == device_location)].copy()

                # if there are any detected non-wear bouts for this device
                if not device_bouts.empty:

                    # get bout indices of wear bouts that meet minimum duration
                    long_wear_idxs = device_bouts.index[(device_bouts['event'] == 'wear')
                                                    & (device_bouts['duration'] >= min_wear_time)]

                    # if there is at least one wear bout of minimum duration
                    if not long_wear_idxs.empty:

                        # select non-wear and wear bouts from first wear of minimum duration to last wear of minimum duration
                        # - same as excluding all wear and non-wear bouts before first and after last wear of minimum duration
                        device_bouts = device_bouts.loc[long_wear_idxs[0]:long_wear_idxs[-1]]

                        # if at least one wear or non-wear bout remains
                        if not device_bouts.empty:

                            # get time info from device data
                            start_time = device.header['start_datetime']

                            # calculate end time of device data
                            samples = len(device.signals[0])
                            sample_rate = device.signal_headers[0]['sample_rate']
                            duration = dt.timedelta(seconds=samples / sample_rate)
                            end_time = start_time + duration

                            # get new start and end time from remaining bouts
                            new_start_time = device_bouts.iloc[0]['start_time']
                            new_end_time = device_bouts.iloc[-1]['end_time']

                            # display messages about duration cropped from start and end of file
                            start_crop_duration = new_start_time - start_time
                            message(f"Cropping {start_crop_duration} from begininng of collection for {device_type} {device_location}",
                                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

                            end_crop_duration = end_time - new_end_time
                            message(f"Cropping {end_crop_duration} from end of collection for {device_type} {device_location}",
                                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

                            # crop device data
                            device.crop(new_start_time, new_end_time, inplace=True)

                    else:
                        message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: Could not crop due to lack of wear time",
                                level='warning', display=(not quiet), log=log, logger_name=self.log_name)

                    # recalculate nonwear summary
                    # nonwear_bouts =  nonwear_bouts_keep[nonwear_bouts_keep.index.isin(nonwear_idx)]
                    db = device_bouts.drop(columns=['study_code', 'subject_id', 'coll_id', 'device_type',
                                                    'device_location', 'accel_std_thresh_mg',
                                                    'low_temperature_cutoff', 'high_temperature_cutoff',
                                                    'temp_dec_roc', 'temp_inc_roc', 'duration'], )
                    daily_nonwear = nonwear_stats(db, quiet=quiet)

                    # save settings used to derive device bouts and add to nonwear dataframe
                    accel_std_thresh_mg = device_bouts.iloc[0]['accel_std_thresh_mg']
                    low_temperature_cutoff = device_bouts.iloc[0]['low_temperature_cutoff']
                    high_temperature_cutoff = device_bouts.iloc[0]['high_temperature_cutoff']
                    temp_dec_roc = device_bouts.iloc[0]['temp_dec_roc']
                    temp_inc_roc = device_bouts.iloc[0]['temp_inc_roc']

                    daily_nonwear.insert(loc=0, column='study_code', value=study_code)
                    daily_nonwear.insert(loc=1, column='subject_id', value=subject_id)
                    daily_nonwear.insert(loc=2, column='coll_id', value=coll_id)
                    daily_nonwear.insert(loc=3, column='device_type', value=device_type)
                    daily_nonwear.insert(loc=4, column='device_location', value=device_location)
                    daily_nonwear.insert(loc=5, column='accel_std_thresh_mg', value=accel_std_thresh_mg)
                    daily_nonwear.insert(loc=6, column='low_temperature_cutoff', value=low_temperature_cutoff)
                    daily_nonwear.insert(loc=7, column='high_temperature_cutoff', value=high_temperature_cutoff)
                    daily_nonwear.insert(loc=8, column='temp_dec_roc', value=temp_dec_roc)
                    daily_nonwear.insert(loc=9, column='temp_inc_roc', value=temp_inc_roc)

                    # update nonwear collection attribrutes
                    coll.daily_nonwear = pd.concat([coll.daily_nonwear, daily_nonwear], ignore_index=True)
                    device_bouts = device_bouts.drop(columns=['duration'])
                    coll.nonwear_bouts = pd.concat([coll.nonwear_bouts, device_bouts], ignore_index=True)
                    coll.devices[i] = device

                else:
                    message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No nonwear data for device",
                            level='warning', display=(not quiet), log=log, logger_name=self.log_name)

            else:
                message(f"{subject_id}_{coll_id}_{device_type}_{device_location}: No nonwear data for collection",
                        level='warning', display=(not quiet), log=log, logger_name=self.log_name)

            # save files
            if save:

                nonwear_csv_name = f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}_NONWEAR.csv"
                daily_nonwear_csv_name = (f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}"
                                          + "_NONWEAR_DAILY.csv")

                nonwear_csv_path = self.dirs['nonwear_bouts_cropped'] / nonwear_csv_name
                nonwear_daily_csv_path = self.dirs['nonwear_daily_cropped'] / daily_nonwear_csv_name

                # check that all folders exist for data output files
                nonwear_csv_path.parent.mkdir(parents=True, exist_ok=True)
                nonwear_daily_csv_path.parent.mkdir(parents=True, exist_ok=True)

                if not coll.nonwear_bouts.empty:

                    # write nonwear times with cropped nonwear removed
                    message(f"Saving {nonwear_csv_path}", level='info', display=(not quiet), log=log,
                            logger_name=self.log_name)
                    device_bouts.to_csv(nonwear_csv_path, index=False)

                    # write new daily non-wear summary
                    message(f"Saving {nonwear_daily_csv_path}", level='info', display=(not quiet), log=log,
                        logger_name=self.log_name)

                    daily_nonwear.to_csv(nonwear_daily_csv_path, index=False)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def save_devices(self, coll, dir, quiet=False, log=True):

        message("Saving device data to EDF...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        for index, row in tqdm(coll.device_info.iterrows(), total=coll.device_info.shape[0], leave=False,
                               desc='Saving device data to EDF'):
            study_code = row['study_code']
            subject_id = row['subject_id']
            coll_id = row['coll_id']
            device_type = row['device_type']
            device_location = row['device_location']
            device_edf_name = f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}.edf"

            # create all file path variables
            device_path = dir / device_edf_name

            # check that all folders exist for data output files
            device_path.parent.mkdir(parents=True, exist_ok=True)

            message(f"Saving {device_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)

            # write device data as edf
            coll.devices[index].export_edf(file_path=device_path, quiet=quiet)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def save_sensors(self, coll, dir, quiet=False, log=True):

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

            device_file_base = f"{study_code}_{subject_id}_{coll_id}_{device_type}_{device_location}"

            # loop through supported sensor types
            for key in tqdm(self.pipeline_settings['pipeline']['sensors'], leave=False, desc="Separating sensors"):

                # search for associated signals in current device
                sig_nums = []
                for sig_label in self.pipeline_settings['pipeline']['sensors'][key]['signals']:
                    sig_num = coll.devices[index].get_signal_index(sig_label)

                    if sig_num is not None:
                        sig_nums.append(sig_num)

                # if signal labels from that sensor are present then save as sensor file
                if sig_nums:

                    sensor_edf_name = '.'.join(['_'.join([device_file_base, key.upper()]), 'edf'])
                    sensor_path = dir / sensor_edf_name
                    sensor_path.parent.mkdir(parents=True, exist_ok=True)

                    message(f"Saving {sensor_path}", level='info', display=(not quiet), log=log,
                            logger_name=self.log_name)

                    coll.devices[index].export_edf(file_path=sensor_path, sig_nums_out=sig_nums, quiet=quiet)

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    @coll_status
    def gait(self, coll, quiet=False, log=True):

        # TODO: axis needs to be set based on orientation of device

        step_detect_type = self.pipeline_settings['modules']['gait']['step_detect_type']
        vert_accel_label = self.pipeline_settings['modules']['gait']['vert_accel']
        sag_gyro_label = self.pipeline_settings['modules']['gait']['sag_gyro']
        save = self.pipeline_settings['modules']['gait']['save']

        message(f"Detecting steps and walking bouts using {step_detect_type} data...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        r_device_idx, l_device_idx = self.select_gait_device(coll=coll)

        if not (l_device_idx or r_device_idx):
            raise NWException(f'{coll.subject_id}_{coll.coll_id}: No left or right ankle device found in device list')

        # TODO: what to do for periods where only one is worn even though both are present
        # TODO: adjust min steps if two or one legs?

        # set indices and handles case if ankle data is missing
        l_device_idx = l_device_idx[0] if l_device_idx else None
        r_device_idx = r_device_idx[0] if r_device_idx else None

        if step_detect_type == 'accel':
            l_sig_label = r_sig_label = vert_accel_label
        elif step_detect_type == 'gyro':
            l_sig_label = r_sig_label = sag_gyro_label
        else:
            message(f"Invalid step_detect_type: {step_detect_type}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            return coll

        if l_device_idx is not None:
            l_sig_idx = coll.devices[l_device_idx].get_signal_index(l_sig_label)
            l_data = coll.devices[l_device_idx].signals[l_sig_idx]
            l_start_time = coll.devices[l_device_idx].header['start_datetime']
            l_fs = coll.devices[l_device_idx].signal_headers[l_sig_idx]['sample_rate']
        else:
            l_data = None
            l_start_time = None
            l_fs = None

        if r_device_idx is not None:
            r_sig_idx = coll.devices[r_device_idx].get_signal_index(r_sig_label)
            r_data = coll.devices[r_device_idx].signals[r_sig_idx]
            r_start_time = coll.devices[r_device_idx].header['start_datetime']
            r_fs = coll.devices[r_device_idx].signal_headers[r_sig_idx]['sample_rate']
        else:
            r_data = None
            r_start_time = None
            r_fs = None

        single_leg = True

        # if two devices
        if (l_data is not None) and (r_data is not None):

            # check that sample rates match
            if l_fs == r_fs:
                    fs=l_fs
            else:
                raise NWException(f'{coll.subject_id}_{coll.coll_id}: Left and right ankle sample rates do not match.')

            # crop data to common start and end time
            start_time = max([l_start_time, r_start_time])

            l_sample_start = int((start_time - l_start_time).total_seconds() * fs)
            l_data = l_data[l_sample_start:]

            r_sample_start = int((start_time - r_start_time).total_seconds() * fs)
            r_data = r_data[r_sample_start:]

            end_sample = min([len(l_data), len(r_data)])

            l_data = l_data[:end_sample]
            r_data = r_data[:end_sample]

            single_leg = False

        elif l_data is not None:
            fs = l_fs
            start_time = l_start_time
        elif r_data is not None:
            fs = r_fs
            start_time = r_start_time

        steps = detect_steps(left_data=l_data, right_data=r_data, loc='ankle', data_type=step_detect_type,
                             start_time=start_time, freq=fs, orient_signal=True, low_pass=12)

        steps, bouts = define_bouts(steps=steps, freq=fs, start_time=start_time, max_break=2, min_steps=3,
                                    remove_unbouted=False)

        coll.gait_steps = steps
        coll.gait_bouts = bouts

        coll.gait_bouts = self.identify_df(coll, coll.gait_bouts)
        coll.gait_steps = self.identify_df(coll, coll.gait_steps)

        message(f"Detected {coll.gait_bouts.shape[0]} gait bouts", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)

        message(f"Detected {coll.gait_steps.shape[0]} steps",
                level='info', display=(not quiet), log=log, logger_name=self.log_name)

        message("Summarizing daily gait analytics...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)

        coll.gait_daily = gait_stats(coll.gait_bouts, stat_type='daily', single_leg=single_leg)
        coll.gait_daily = self.identify_df(coll, coll.gait_daily)


        bout_cols = ['study_code', 'subject_id', 'coll_id', 'gait_bout_num', 'start_time', 'end_time',
                     'step_count']
        coll.gait_bouts = coll.gait_bouts[bout_cols]

        step_cols = ['study_code','subject_id','coll_id','step_num', 'gait_bout_num', 'step_time', 'step_idx', 'loc',
                     'side', 'data_type', 'alg']
        coll.gait_steps = coll.gait_steps[step_cols]

        if save:
            # create all file path variables
            bouts_csv_name = f"{coll.study_code}_{coll.subject_id}_{coll.coll_id}_GAIT_BOUTS.csv"
            steps_csv_name = f"{coll.study_code}_{coll.subject_id}_{coll.coll_id}_GAIT_STEPS.csv"
            daily_gait_csv_name = f"{coll.study_code}_{coll.subject_id}_{coll.coll_id}_GAIT_DAILY.csv"

            bouts_csv_path = self.dirs['gait_bouts'] / bouts_csv_name
            steps_csv_path = self.dirs['gait_steps'] / steps_csv_name
            daily_gait_csv_path = self.dirs['gait_daily'] / daily_gait_csv_name

            message(f"Saving {bouts_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.log_name)
            coll.gait_bouts.to_csv(bouts_csv_path, index=False)

            message(f"Saving {steps_csv_path}", level='info', display=(not quiet), log=log, logger_name=self.log_name)
            coll.gait_steps.to_csv(steps_csv_path, index=False)

            message(f"Saving {daily_gait_csv_path}", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            coll.gait_daily.to_csv(daily_gait_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def read_gait(self, coll, single_stage, quiet=False, log=True):

        # read gait data for all devices
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

        save = self.pipeline_settings['modules']['sleep']['save']

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
        sleep_days = 0
        if 'long' in daily_sleep_t5a5['sptw_inc']:
            sleep_days = daily_sleep_t5a5['sptw_inc'].value_counts()['long']
        message(f"Summarized {sleep_days} days of sleep (t5a5)...",
                level='info', display=(not quiet), log=log, logger_name=self.log_name)

        daily_sleep_t8a4 = sptw_stats(coll.sptw, sleep_t8a4, type='daily', sptw_inc=['long', 'all', 'sleep', 'overnight_sleep'])
        sleep_days = 0
        if 'long' in daily_sleep_t8a4['sptw_inc']:
            sleep_days = daily_sleep_t8a4['sptw_inc'].value_counts()['long']
        message(f"Summarized {sleep_days} days of sleep analytics (t8a4)...",
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

        pref_cutpoint = self.pipeline_settings['modules']['activity'].get('pref_cutpoint', None)
        save = self.pipeline_settings['modules']['activity']['save']
        epoch_length = self.pipeline_settings['modules']['activity']['epoch_length']
        sedentary_gait = self.pipeline_settings['modules']['activity']['sedentary_gait']

        dominant_hand = coll.collection_info['dominant_hand'].lower()

        # select all wrist devices
        activity_device_index = self.select_activity_device(coll=coll, quiet=quiet, log=log)

        if len(activity_device_index) == 0:
            raise NWException(f"{coll.subject_id}_{coll.coll_id}: No eligible wrist devices found in device list")


        for c, i in enumerate(activity_device_index):

            # checks to see if data exists
            if not coll.devices[i]:
                message(f'{coll.subject_id}_{coll.coll_id}: Wrist device data is missing', level='warning',
                        display=(not quiet), log=log, logger_name=self.log_name)
                message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)
                continue

            device_type = coll.device_info.loc[i]['device_type']
            device_location = coll.device_info.loc[i]['device_location']

            accel_x_sig = coll.devices[i].get_signal_index('Accelerometer x')
            accel_y_sig = coll.devices[i].get_signal_index('Accelerometer y')
            accel_z_sig = coll.devices[i].get_signal_index('Accelerometer z')

            message(f"Calculating {epoch_length}-second epoch activity for {device_type}_{device_location}...",
                    level='info', display=(not quiet), log=log, logger_name=self.log_name)

            cutpoint_ages = pd.DataFrame(self.pipeline_settings['modules']['activity']['cutpoints'])

            subject_age = int(coll.collection_info['age'])
            lowpass = int(self.pipeline_settings['modules']['activity']['lowpass'])

            cutpoint = cutpoint_ages['type'].loc[(cutpoint_ages['min_age'] <= subject_age)
                                                 & (cutpoint_ages['max_age'] >= subject_age)].item()

            # select dominant or non-dominant cutpoint
            if pref_cutpoint == "dominant":
                dominant = True
            elif pref_cutpoint == "non-dominant":
                dominant = False
            else:
                if dominant_hand in ['right', 'left']:
                    dominant_wrist = dominant_hand[0] + 'wrist'
                    dominant = device_location.upper() in self.pipeline_settings['pipeline']['device_locations'][dominant_wrist]['aliases']
                else:
                    dominant = True


            # get nonwear for activity_device
            device_nonwear = coll.nonwear_bouts.loc[(coll.nonwear_bouts['study_code'] == coll.study_code) &
                                                    (coll.nonwear_bouts['subject_id'] == coll.subject_id) &
                                                    (coll.nonwear_bouts['coll_id'] == coll.coll_id) &
                                                    (coll.nonwear_bouts['device_type'] == device_type) &
                                                    (coll.nonwear_bouts['device_location'] == device_location) &
                                                    (coll.nonwear_bouts['event'] == 'nonwear')]

            sptw = coll.sptw
            if coll.sleep_bouts.empty:
                sleep_bouts = pd.DataFrame()
            else:
                sleep_bouts =  coll.sleep_bouts.loc[coll.sleep_bouts['bout_detect'] == 't8a4']

            e, b, avm, vm, avm_sec = activity_wrist_avm(x=coll.devices[i].signals[accel_x_sig],
                                                        y=coll.devices[i].signals[accel_y_sig],
                                                        z=coll.devices[i].signals[accel_z_sig],
                                                        sample_rate=coll.devices[i].signal_headers[accel_x_sig]['sample_rate'],
                                                        start_datetime=coll.devices[i].header['start_datetime'],
                                                        lowpass=lowpass, epoch_length=epoch_length, cutpoint=cutpoint,
                                                        dominant=dominant, sedentary_gait=sedentary_gait,
                                                        gait=coll.gait_bouts, nonwear=device_nonwear, sptw=sptw,
                                                        sleep_bouts=sleep_bouts, quiet=quiet)

            activity_epochs = e
            activity_bouts = b

            # prepare avm dataframe
            avm_second = pd.DataFrame()
            avm_second['avm_num'] = np.arange(1, len(avm_sec) + 1)
            avm_second['avm'] = avm_sec
            avm_second.insert(loc=0, column='device_location', value=device_location)
            avm_second = self.identify_df(coll, avm_second)

            message("Summarizing daily activity volumes...", level='info', display=(not quiet), log=log,
                    logger_name=self.log_name)
            activity_daily = activity_stats(activity_bouts, quiet=quiet)

            activity_epochs.insert(loc=1, column='device_location',value=device_location)
            activity_epochs.insert(loc=2, column='cutpoint_type', value=cutpoint)
            activity_epochs.insert(loc=3, column='cutpoint_dominant', value=dominant)

            activity_bouts.insert(loc=1, column='device_location', value=device_location)
            activity_bouts.insert(loc=2, column='cutpoint_type', value=cutpoint)
            activity_bouts.insert(loc=3, column='cutpoint_dominant', value=dominant)

            activity_epochs = self.identify_df(coll, activity_epochs)
            activity_bouts = self.identify_df(coll, activity_bouts)

            activity_daily.insert(loc=2, column='device_location', value=device_location)
            activity_daily.insert(loc=3, column='cutpoint_type', value=cutpoint)
            activity_daily.insert(loc=4, column='cutpoint_dominant', value=dominant)
            activity_daily.insert(loc=5, column='type', value='daily')

            activity_daily = self.identify_df(coll, activity_daily)

            if c == 0:
                coll.activity_epochs = activity_epochs
                coll.activity_bouts = activity_bouts
                coll.activity_daily = activity_daily
                coll.avm_second = avm_second
            else:
                coll.activity_epochs = pd.concat([coll.activity_epochs, activity_epochs])
                coll.activity_bouts = pd.concat([coll.activity_bouts, activity_bouts])
                coll.activity_daily = pd.concat([coll.activity_daily, activity_daily])
                coll.avm_second = pd.concat([coll.avm_second, avm_second])

            message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

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
            coll.avm_second.to_csv(avm_csv_path, index=False)

        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        return coll

    def collection_report(self, coll, quiet=False, log=True):

        message("Creating collection report...", level='info', display=(not quiet), log=log,
                logger_name=self.log_name)
        message("", level='info', display=(not quiet), log=log, logger_name=self.log_name)

        include_supp = self.pipeline_settings['modules']['collection_report']['include_supp']
        include_custom = self.pipeline_settings['modules']['collection_report']['include_custom']
        daily_plot = self.pipeline_settings['modules']['collection_report']['daily_plot']
        fig_size = tuple(self.pipeline_settings['modules']['collection_report']['fig_size'])
        top_y = tuple(self.pipeline_settings['modules']['collection_report']['top_y'])
        bottom_y = tuple(self.pipeline_settings['modules']['collection_report']['bottom_y'])
        supp_path = self.pipeline_settings['modules']['collection_report'].get('supp_path', None)


        cr(study_dir=self.study_dir, subject_id=coll.subject_id, coll_id=coll.coll_id, supp_pwd=coll.supp_pwd,
           include_supp=include_supp, include_custom=include_custom, daily_plot=daily_plot, fig_size=fig_size,
           top_y=top_y, bottom_y=bottom_y, supp_path=supp_path)

        return coll

    def select_activity_device(self, coll, quiet=False, log=True):

        # select devices to use for activity level
        device_info_copy = coll.device_info.copy()

        # convert device location to upper case
        device_info_copy['device_location'] = [x.upper() for x in device_info_copy['device_location']]

        # select eligible device types and locations
        activity_device_types = ['GNOR', 'AXV6']
        activity_locations = (self.pipeline_settings['pipeline']['device_locations']['rwrist']['aliases']
                              + self.pipeline_settings['pipeline']['device_locations']['lwrist']['aliases'])

        # get index of all eligible devices based on type and location
        activity_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(activity_device_types)) &
                                                     (device_info_copy['device_location'].isin(activity_locations))].index.values.tolist()

        # select device from list based on wrist preference
        # if (pref_wrist != 'all') & (len(activity_device_index) > 1):
        #
        #     # select dominant or non-dominant based on argument
        #     if (pref_wrist == 'dominant') & (dominant_hand in ['right', 'left']):
        #             pref_wrist = dominant_hand
        #     elif (pref_wrist == 'non-dominant') & (dominant_hand in ['right', 'left']):
        #             pref_wrist = {'left': "right", 'right': "left"}[dominant_hand]
        #
        #     # if no dominant hand info display warning and take first device
        #     if pref_wrist in ['dominant', 'non-dominant']:
        #         message(f"Preferred wrist is {pref_wrist} but no dominant hand info found - selecting first eligible device...",
        #                 level='warning', display=(not quiet), log=log, logger_name=self.log_name)
        #         activity_device_index = [activity_device_index[0]]
        #
        #     else:
        #
        #         wrist = pref_wrist[0] + 'wrist'
        #
        #         # select devices at locations based on dominance
        #         activity_locations = self.device_locations[wrist]['aliases']
        #         activity_device_index = device_info_copy.loc[
        #             (device_info_copy['device_type'].isin(activity_device_types)) &
        #             (device_info_copy['device_location'].isin(activity_locations))].index.values.tolist()
        #
        #         # if still multiple eligible devices, take first one
        #         if len(activity_device_index) > 1:
        #             activity_device_index = [activity_device_index[0]]
        #
        #         # if no eligible devices, go back and take first one from list of all eligible
        #         elif len(activity_device_index) < 1:
        #             activity_locations = self.device_locations['rwrist']['aliases'] + self.device_locations['lwrist']['aliases']
        #             activity_device_index = device_info_copy.loc[
        #                 (device_info_copy['device_type'].isin(activity_device_types)) &
        #                 (device_info_copy['device_location'].isin(activity_locations))].index.values.tolist()
        #             activity_device_index = [activity_device_index[0]]

        # # if only one device determine, if it is dominant
        # elif len(activity_device_index) == 1:
        #
        #     # if dominant hand info is available we will determine dominance
        #     if dominant_hand in ['right', 'left']:
        #         dominant_wrist = dominant_hand[0] + 'wrist'
        #         dominant = device_info_copy.loc[activity_device_index]['device_location'].item() in \
        #                    self.device_locations[dominant_wrist]['aliases']

            # if no dominant hand info available, assume dominant argument is correct

        return activity_device_index

    def select_gait_device(self, coll):

        device_info_copy = coll.device_info.copy()
        device_info_copy['device_location'] = [x.upper() for x in device_info_copy['device_location']]

        # select eligible device types and locations
        gait_device_types = ['GNOR', 'AXV6']
        r_gait_locations = self.pipeline_settings['pipeline']['device_locations']['rankle']['aliases']
        l_gait_locations = self.pipeline_settings['pipeline']['device_locations']['lankle']['aliases']

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

        dominant = self.pipeline_settings['modules']['sleep']['dominant']
        dominant_hand = coll.collection_info['dominant_hand'].lower()

        device_info_copy = coll.device_info.copy()
        device_info_copy['device_location'] = [x.upper() for x in device_info_copy['device_location']]

        # select eligible device types and locations
        sleep_device_types = ['GNOR', 'AXV6']
        sleep_locations = (self.pipeline_settings['pipeline']['device_locations']['rwrist']['aliases']
                           + self.pipeline_settings['pipeline']['device_locations']['lwrist']['aliases'])

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
                sleep_locations = self.pipeline_settings['pipeline']['device_locations'][wrist]['aliases']
                sleep_device_index = device_info_copy.loc[(device_info_copy['device_type'].isin(sleep_device_types)) &
                                                          (device_info_copy['device_location'].isin(sleep_locations))].index.values.tolist()

                # if still multiple eligible devices, take first one
                if len(sleep_device_index) > 1:
                    sleep_device_index = [sleep_device_index[0]]

                # if no eligible devices, go back and take first one from list of all eligible
                elif len(sleep_device_index) < 1:
                    sleep_locations = (self.pipeline_settings['pipeline']['device_locations']['rwrist']['aliases']
                                       + self.pipeline_settings['pipeline']['device_locations']['lwrist']['aliases'])
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
                dominant = device_info_copy.loc[sleep_device_index]['device_location'].item() in self.pipeline_settings['pipeline']['device_locations'][dominant_wrist]['aliases']

            # if no dominant hand info available, assume dominant argument is correct

        return sleep_device_index, dominant

    def add_custom_events(self, file_path, quiet=False):
        """Import properly formatted csv of events.

        Parameters
        ----------
        file_path : str or Path
            Path to properly formatted csv of new events.
        quiet : bool, optional
            Suppress displayed messages (default is False)

        """

        file_path = Path(file_path)

        # read new events from csv
        index_cols = ['study_code', 'subject_id', 'coll_id', 'event', 'id', ]
        dtype_cols = {'study_code': str, 'subject_id': str, 'coll_id': str, 'event': str, 'id': pd.Int64Dtype(),
                      'details': str,
                      'notes': str, }
        date_cols = ['start_time', 'end_time', ]

        if not file_path.is_file():
            print(f"{file_path} does not exist.")
            return False

        if not quiet:
            print(f"Reading new events file: {file_path}\n")

        new_events = pd.read_csv(file_path, index_col=index_cols, dtype=dtype_cols, parse_dates=date_cols)

        # ensure new events have unique index
        if not new_events.index.is_unique:
            print("Events could not be added because some events could not be uniquely identified by study_code, "
                  "subject_id, coll_id, event, id columns.\n")
            return False

        # ensure start_time is not blank
        if any(new_events['start_time'].isnull()):
            print("Events could not be added because start_time is required and some were blank.\n")
            return False

        # ensure study code of all events matches
        if any(new_events.index.get_level_values('study_code') != self.study_code):
            print("Events could not be added because some study codes did not match current study.\n")
            return False

        custom_events_dir = self.dirs['events_custom']

        unique_new_collections = new_events.reset_index().set_index(['study_code', 'subject_id', 'coll_id']).index.unique()

        # loop through unique collections in new events file
        for collection in unique_new_collections:

            # get events for this collection only
            new_collection_events = new_events.loc[([collection[0]], [collection[1]], [collection[2]])]

            # generate custom events csv path
            events_csv_name = f"{collection[0]}_{collection[1]}_{collection[2]}_EVENTS_CUSTOM.csv"
            events_csv_path = custom_events_dir / events_csv_name

            if events_csv_path.is_file():       # custom events csv already exists

                # read csv
                if not quiet:
                    print(f"Reading custom events file: {events_csv_path}")
                events = pd.read_csv(events_csv_path, index_col=index_cols, dtype=dtype_cols, parse_dates=date_cols)

                # determine new event types being added and remove any events of those type that already exist
                # - this is done to avoid confusion within a type of event - best to remove all of a type and re-add them
                new_event_types = new_collection_events.index.unique('event').values

                if not quiet:
                    print(f"Replacing or adding events of following types: {', '.join(new_event_types)}.")

                events = events[~events.index.get_level_values('event').isin(new_event_types)]

                # add new events
                events = pd.concat([events, new_collection_events])

            else:       # custom events csv doesn't exist

                events = new_collection_events

            # save custom events csv
            if not quiet:
                print(f"Saving {events_csv_path}\n")

            events = events.sort_values(by='start_time')
            events_csv_path.parent.mkdir(parents=True, exist_ok=True)
            events.to_csv(events_csv_path)

        return True

    def identify_df(self, coll, df):
        df.insert(loc=0, column='study_code', value=self.study_code)
        df.insert(loc=1, column='subject_id', value=coll.subject_id)
        df.insert(loc=2, column='coll_id', value=coll.coll_id)
        return df

    def get_collections(self):

        collections = [(row['subject_id'], row['coll_id']) for i, row in self.collection_info.iterrows()]

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

        self.nonwear_bouts = pd.DataFrame()
        self.daily_nonwear = pd.DataFrame()

        self.gait_steps = pd.DataFrame()
        self.gait_bouts = pd.DataFrame()
        self.gait_daily = pd.DataFrame()

        self.sptw = pd.DataFrame()
        self.sleep_bouts = pd.DataFrame()
        self.daily_sleep = pd.DataFrame()

        self.activity_epochs = pd.DataFrame()
        self.activity_bouts = pd.DataFrame()
        self.activity_daily = pd.DataFrame()
        self.avm_second = pd.DataFrame()


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