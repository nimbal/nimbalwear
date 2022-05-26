"""Data module.

This module contains the Data class, which represents wearable device data
in a structure compatible with the European Data Format.

"""

# TODO: detailed transducer info for all devices?
# TODO: Return true or false depending on method success?
# TODO: implement quiet for all methods

import datetime as dt
import math
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from .files import EDFFile, GENEActivFile, NoninFile, CWAFile
from .utils import sync_devices, autocal


class Data:
    """A class used to represent wearable device data.

    The structure of this class is based on the European Data Format (EDF)
    file, which is the default standardized file used to store wearable device
    data in the NiMBalWear data processing pipeline. This class contains
    methods for importing data from various file types, deidentifying and
    cropping data, and exporting data to EDF files.

    Attributes:
        header (dict): Dictionary containing all attributes from the
            file header
        signal_headers (list): List containing a dictionary with header
            attributes for each signal
        signals (list): List of arrays of data for each signal

    """

    def __init__(self):
        """ Initialize attributes on object construction."""

        # initialize attributes
        self.header = {'study_code': '',
                       'subject_id': '',
                       'coll_id': '',
                       'name': '',
                       'sex': '',
                       'birthdate': '',
                       'start_datetime': '',
                       'config_datetime': '',
                       'technician': '',
                       'device_type': '',
                       'device_id': '',
                       'device_location': '',
                       'recording_additional': ''}      # device_location config_date config_time

        self.signal_headers = []
        self.signals = []

    def import_edf(self, file_path, quiet=False):
        """Imports data from an EDF file.

        Args:
            file_path (str): Full absolute path to the file to be imported
            quiet (bool):
        Returns:
            True if successful, False otherwise.
        """

        file_path = Path(file_path)

        # check if file exists
        if not file_path.is_file():
            print("Import failed: file does not exist.")
            return False

        # read file
        edf_file = EDFFile(file_path)
        edf_file.read(quiet=quiet)

        p_add = [item for item in edf_file.header['patient_additional'].split(' ') if item != '']
        r_add = [item for item in edf_file.header['recording_additional'].split(' ') if item != '']

        coll_id = p_add[0] if len(p_add) > 0 else ''
        device_location = p_add[1] if len(p_add) > 1 else ''

        config_datetime = r_add[0] if len(r_add) > 0 else ''
        config_datetime = pd.to_datetime(config_datetime, format='%Y%m%d%H%M%S', errors='coerce')

        equipment = [item for item in edf_file.header['equipment'].split('_') if item != '']
        device_type = equipment[0] if len(equipment) > 0 else ''
        device_id = equipment[1] if len(equipment) > 1 else ''

        self.header = {'study_code': edf_file.header['admincode'],
                       'subject_id': edf_file.header['patientcode'],
                       'coll_id': coll_id,
                       'name': edf_file.header['patientname'],
                       'sex': edf_file.header['gender'],  # 0=F, 1=M, 2=X
                       'birthdate': edf_file.header['birthdate'],
                       'patient_additional': ' '.join(p_add[2:]) if len(p_add) > 2 else '',
                       'start_datetime': edf_file.header['startdate'],
                       'config_datetime': config_datetime,
                       'technician': edf_file.header['technician'],
                       'device_type': device_type,
                       'device_id': device_id,
                       'device_location': device_location,
                       'recording_additional': ' '.join(r_add[1:]) if len(r_add) > 1 else ''}

        self.signal_headers = edf_file.signal_headers
        self.signals = edf_file.signals

        return True

    def export_edf(self, file_path, sig_nums_out=None, quiet=False):
        """Exports data to an EDF file.

        Args:
            file_path (str): Full absolute path to the file to be written.
            sig_nums_out (str, optional): Signal numbers to export, defaults
                to None signifying all signals will be exported.
            quiet (bool, optional): Suppress messages, defaults to False.

        Returns:
            True if successful, False otherwise.
        """

        file_path = Path(file_path)

        # error handling
        error_str = "Export failed: "

        # check for signal data
        if not self.signals:
            print(error_str + "No signal data.")
            return False

        # check that signal headers match signals
        if not len(self.signal_headers) == len(self.signals):
            print(error_str + "number of signal headers does not match number of signals.")
            return False

        # check that startdate is a datetime - indicates header has been read in
        if not isinstance(self.header['start_datetime'], dt.datetime):
            print(error_str + "start_datetimme is not a datetime object - header may not have been imported correctly.")
            return False

        # ensure valid sig nums used as arguments
        if sig_nums_out is not None:
            if any([sig_num >= len(self.signals) for sig_num in sig_nums_out]):
                print(error_str + "invalid signal number.")
                return False

        # check that all folders exist for data output file
        file_path.parent.mkdir(parents=True, exist_ok=True)

        config_datetime = (self.header['config_datetime'].strftime('%Y%m%d%H%M%S')
                           if pd.notnull(self.header['config_datetime']) else '')

        # write to edf
        edf_file = EDFFile(file_path)

        edf_file.header = {'patientcode': self.header['subject_id'],
                           'gender': self.header['sex'],
                           'birthdate': self.header['birthdate'],
                           'patientname': self.header['name'],
                           'patient_additional': ' '.join([self.header['coll_id'], self.header['device_location']]),
                           'startdate': self.header['start_datetime'],
                           'admincode': self.header['study_code'],
                           'technician': self.header['technician'],
                           'equipment': '_'.join([self.header['device_type'], self.header['device_id']]),
                           'recording_additional': config_datetime}

        for i, s_h in enumerate(self.signals):
            self.signal_headers[i]['physical_max'] = max(self.signal_headers[i]['physical_max'], max(self.signals[i]))
            self.signal_headers[i]['physical_min'] = min(self.signal_headers[i]['physical_min'], min(self.signals[i]))
            self.signal_headers[i]['digital_max'] = 32767
            self.signal_headers[i]['digital_min'] = -32768
            self.signal_headers[i]['prefilter'] = ''

        edf_file.signal_headers = self.signal_headers
        edf_file.signals = self.signals
        edf_file.write(file_path, sig_nums_out=sig_nums_out, quiet=quiet)

        return True

    def deidentify(self):
        """Removes information from header fields that may identify individual.

        Returns:
            True if successful, False otherwise.
        """

        # deidentify data by blanking out gender, birthdate, and patientname
        self.header.update({'sex': '', 'birthdate': '', 'name': ''})

        return True

    def get_signal_index(self, label):

        index = 0

        for signal_header in self.signal_headers:
            if signal_header['label'].strip() == label:
                break
            else:
                index += 1

        index = None if index == len(self.signal_headers) else index

        return index

    def get_day_idxs(self, day_offset):

        start_datetime = self.header['start_datetime']

        day_start_times = []
        day_start_idxs = []

        for idx, sig_head in enumerate(self.signal_headers):

            sample_rate = sig_head['sample_rate']
            num_samples = len(self.signals[idx])

            first_day_start = dt.datetime.combine(start_datetime - dt.timedelta(hours=day_offset),
                                                  dt.time.min) + dt.timedelta(hours=day_offset)
            days = ((num_samples / sample_rate) + (start_datetime - first_day_start).total_seconds()) / (60 * 60 * 24)

            sig_day_start_times = [first_day_start + dt.timedelta(days=x) for x in range(math.ceil(days))]
            sig_day_start_times[0] = start_datetime

            sig_day_start_idxs = [round((day_start_time - start_datetime).total_seconds() * sample_rate) for day_start_time
                                     in sig_day_start_times]

            day_start_times.append(sig_day_start_times)
            day_start_idxs.append(sig_day_start_idxs)

        return day_start_times, day_start_idxs

    def get_idxs_from_date(self, date):

        start_datetime = self.header['start_datetime']

        idxs = []

        for idx, sig_head in enumerate(self.signal_headers):
            sample_rate = sig_head['sample_rate']

            idxs.append(round((date - start_datetime).total_seconds() * sample_rate))

        return idxs

    def crop(self, new_start_time=None, new_end_time=None, inplace=False):
        """Crops data from start or end of all signals.

        Args:
            new_start_time (datetime, optional): New start time of cropped data,
                defaults to None indicating no cropping from start of data.
            new_end_time (datetime, optional): New end time of cropped data,
                defaults to None indicating no cropping from end of data.
            inplace (boolean, optional): Indicates whether current object is modified or a cropped copy is returned,
                defaults to False indicating that a cropped copy of the object is returned.

        Returns:
            True if successful, False otherwise.
        """

        # check to see if data exists
        error_str = "Crop failed: "

        # check that header, signal_headers, signals exist and are same length, etc.
        if not self.signals:
            print(error_str + "No signal data.")
            return False

        if not len(self.signal_headers) == len(self.signals):
            print(error_str + "number of signal headers does not match number of signals.")
            return False

        if not isinstance(self.header['start_datetime'], dt.datetime):
            print(error_str + "startdate is not a datetime object - header may not have been imported correctly.")
            return False

        # read or calculate collection time info
        start_time = self.header['start_datetime']
        duration = dt.timedelta(seconds=len(self.signals[0]) / self.signal_headers[0]['sample_rate'])
        end_time = start_time + duration

        # set crop times based on input arguments
        if new_start_time is None:
            new_start_time = start_time
        if new_end_time is None:
            new_end_time = end_time
        if new_end_time < new_start_time:
            new_end_time = end_time
        if new_start_time < start_time:
            new_start_time = start_time
        if new_end_time > end_time:
            new_end_time = end_time

        if inplace:
            new_self = self
        else:
            new_self = deepcopy(self)

        new_self.header['start_datetime'] = new_start_time

        # loop through signals
        for sig_num in range(len(new_self.signal_headers)):

            # calculate new start, duration, and end samples
            crop_start = int((new_start_time - start_time).total_seconds()
                             * new_self.signal_headers[sig_num]['sample_rate'])
            crop_duration = int((new_end_time - new_start_time).total_seconds()
                                * new_self.signal_headers[sig_num]['sample_rate'])
            crop_end = crop_start + crop_duration

            # crop signal
            new_self.signals[sig_num] = new_self.signals[sig_num][crop_start:crop_end]

        if inplace:
            return True
        else:
            return new_self

    def rotate_z(self, deg):
        """
        Rotate the accelerometer and/or gyroscope by deg
        Args:
            deg: degrees to rotate
        """

        r = R.from_euler(seq='z', angles=deg, degrees=True)

        ax_idx = self.get_signal_index("Accelerometer x")
        ay_idx = self.get_signal_index("Accelerometer y")
        az_idx = self.get_signal_index("Accelerometer z")
        gx_idx = self.get_signal_index("Gyroscope x")
        gy_idx = self.get_signal_index("Gyroscope y")
        gz_idx = self.get_signal_index("Gyroscope z")

        accel_idx = [ax_idx, ay_idx, az_idx]
        gyro_idx = [gx_idx, gy_idx, gz_idx]

        if None not in accel_idx:
            accel = np.array([self.signals[i] for i in accel_idx]).transpose()
            rot_accel = r.apply(accel).transpose()
            self.signals[ax_idx] = rot_accel[0].tolist()
            self.signals[ay_idx] = rot_accel[1].tolist()
            self.signals[az_idx] = rot_accel[2].tolist()

        if None not in gyro_idx:
            gyro = np.array([self.signals[i] for i in gyro_idx]).transpose()
            rot_gyro = r.apply(gyro).transpose()
            self.signals[gx_idx] = rot_gyro[0].tolist()
            self.signals[gy_idx] = rot_gyro[1].tolist()
            self.signals[gz_idx] = rot_gyro[2].tolist()

        return True

    def autocal(self, use_temp=True, epoch_secs=None, detect_only=False, plot=False, quiet=False):

        # get accelerometer x, y, z and temperature signals
        x_i = self.get_signal_index('Accelerometer x')
        y_i = self.get_signal_index('Accelerometer y')
        z_i = self.get_signal_index('Accelerometer z')

        temp = None
        temp_fs = None

        if use_temp:
            temp_i = self.get_signal_index('Temperature')
            temp = self.signals[temp_i]
            temp_fs = self.signal_headers[temp_i]['sample_rate']

        x, y, z, pre_err, post_err, iter = autocal(x=self.signals[x_i], y=self.signals[y_i], z=self.signals[z_i],
                                                   accel_fs=self.signal_headers[x_i]['sample_rate'], temp=temp,
                                                   temp_fs=temp_fs, use_temp=use_temp, epoch_secs=epoch_secs,
                                                   detect_only=detect_only, plot=plot, quiet=quiet)

        self.signals[x_i] = x
        self.signals[y_i] = y
        self.signals[z_i] = z

        return pre_err, post_err, iter

    def sync(self, ref, sig_labels=('Accelerometer x', 'Accelerometer y', 'Accelerometer z'), type='flip',
             sync_at_config=True, **kwargs):

        syncs = None
        segments = None

        if type == 'flip':

            ref_config_time = ref.header['config_datetime']

            last_sync = ref_config_time if sync_at_config and ref_config_time < ref.header['start_datetime'] else None

            # Add warning if config time > start_timme

            syncs, segments = sync_devices(self, ref, sig_labels=sig_labels, last_sync=last_sync, **kwargs)

        else:

            print('Invalid sync type')

        return syncs, segments

    def import_bittium(self, file_path, quiet=False):
        """Imports data from a Bittium-device generated EDF file.

        This differs from a regular EDF file because it reformats the header
        to match the style expected by NiMBaLWear pipeline.

        Args:
            file_path (str): Full absolute path to the file to be imported.
            quiet (bool):

        Returns:
            True if successful, False otherwise.
        """

        file_path = Path(file_path)

        # check if file exists
        if not file_path.is_file():
            print("Import failed: file does not exist.")
            return False

        # read Bittium edf file
        in_file = EDFFile(file_path)
        in_file.read(quiet=quiet)
        self.signal_headers = in_file.signal_headers
        self.signals = in_file.signals

        patient_code = in_file.header['patientcode'].split('_')
        study_code = patient_code[0] if len(patient_code) else ''
        subject_id = patient_code[1] if len(patient_code) > 1 else ''
        coll_id = patient_code[2] if len(patient_code) > 2 else ''

        # get serial id and device type
        serial_id = in_file.header['equipment'].split('SER=', 1)[1].split('_', 1)[0]
        device_type = in_file.header['equipment'].split('DEVTYPE=', 1)[1].split('_', 1)[0]
        if device_type == 'Faros360':
            device_type = 'BF36'
        elif device_type == 'Faros180':
            device_type = 'BF18'
        else:
            device_type = 'BFXX'

        self.header = {'study_code': study_code,
                       'subject_id': subject_id,
                       'coll_id': coll_id,
                       'name': in_file.header['patientname'],
                       'sex': in_file.header['gender'],
                       'birthdate': in_file.header['birthdate'],
                       'patient_additional': '',
                       'start_datetime': in_file.header['startdate'],
                       'config_datetime': pd.NaT,
                       'technician': in_file.header['technician'],
                       'device_type': device_type,
                       'device_id': serial_id,
                       'device_location': 'Chest',
                       'recording_additional': ''}

        # update signal headers to match nwdata
        new_signal_headers = {'ECG': {'label': "ECG", 'transducer': "ECG"},
                              'Accelerometer_X': {'label': "Accelerometer x", 'transducer': "Accelerometer",
                                                  'dimension': "g"},
                              'Accelerometer_Y': {'label': "Accelerometer y", 'transducer': "Accelerometer",
                                                  'dimension': "g"},
                              'Accelerometer_Z': {'label': "Accelerometer z", 'transducer': "Accelerometer",
                                                  'dimension': "g"},
                              'HRV': {'label': "HRV", 'transducer': "ECG"},
                              'DEV_Temperature': {'label': "Temperature", 'transducer': "Thermometer"}}

        for key in new_signal_headers:
            sig_ind = self.get_signal_index(key)
            if sig_ind is not None:
                self.signal_headers[sig_ind].update(new_signal_headers[key])

        # convert accelerometer signals from mg to g
        sig_labels = ['Accelerometer x', 'Accelerometer y', 'Accelerometer z']
        for label in sig_labels:
            idx = self.get_signal_index(label)
            self.signal_headers[idx]['physical_min'] = self.signal_headers[idx]['physical_min'] * 0.001
            self.signal_headers[idx]['physical_max'] = self.signal_headers[idx]['physical_max'] * 0.001
            self.signals[idx] = self.signals[idx] * 0.001

        return True

    def import_geneactiv(self, file_path, parse_data=True, start=1, end=-1, downsample=1, calibrate=True,
                         correct_drift=True, quiet=False):
        """Imports data from a GENEActiv-device-generated binary file.


        Args:
            file_path (str): Full absolute path to the file to be imported.
            parse_data (bool): Parse hexadecimal data to decimal and store,
                default is True.
            start (int): Page at which to start parsing data, default is 1.
            end (int): Page at which to end parsing data, default is -1
                indicating the final page.
            downsample (int): Factor by which to downsample, default is 1
                indicating no downsampling.
            calibrate (bool): Convert stored digital values to physical values
                based on stored offset and gain, default is True.
            correct_drift (bool): Correct clock drift, default is True.
            quiet (bool): Suppress messages, default is False.

        Returns:
            True if successful, False otherwise.
        """

        file_path = Path(file_path)

        # check if file exists
        if not file_path.is_file():
            print("Import failed: file does not exist.")
            return False

        # read file
        in_file = GENEActivFile(file_path)
        in_file.read(parse_data=parse_data, start=start, end=end, downsample=downsample,
                     calibrate=calibrate, correct_drift=correct_drift, quiet=quiet)

        # find amount to trim from start to shift to next second integer (EDF files don't store start time to ms)
        trim_microseconds = (1000000 - in_file.data['start_time'].microsecond
                             if in_file.data['start_time'].microsecond > 0
                             else 0)
        trim_samples = round(in_file.data['sample_rate'] * (trim_microseconds / 1000000))

        # update header to match nwdata format
        self.header = {'study_code': in_file.header['Study Code'],
                       'subject_id': in_file.header['Subject Code'],
                       'coll_id': in_file.header['Subject Notes'],
                       'name': '',
                       'sex': in_file.header['Sex'],
                       'birthdate': pd.to_datetime(in_file.header["Date of Birth"], format="%Y-%m-%d"),
                       'patient_additional': '',
                       'start_datetime': in_file.data['start_time'] + dt.timedelta(microseconds=trim_microseconds),
                       'config_datetime': pd.to_datetime(in_file.header['Config Time'], format='%Y-%m-%d %H:%M:%S:%f'),
                       'technician': in_file.header['Investigator ID'],
                       'device_type': 'GNOR',
                       'device_id': in_file.header['Device Unique Serial Code'],
                       'device_location': in_file.header['Device Location Code'].replace(' ', '_'),
                       'recording_additional': ''}

        # update signal headers to match nwdata format
        self.signal_headers = [{'label': "Accelerometer x",
                                'transducer': "MEMS Accelerometer",
                                'dimension': in_file.header['Accelerometer Units'],
                                'sample_rate': in_file.data['sample_rate'],
                                'physical_max': ((204700 - int(in_file.header['x offset']))
                                                 / int(in_file.header['x gain'])),
                                'physical_min': ((-204800 - int(in_file.header['x offset']))
                                                 / int(in_file.header['x gain'])),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Accelerometer y",
                                'transducer': "MEMS Accelerometer",
                                'dimension': in_file.header['Accelerometer Units'],
                                'sample_rate': in_file.data['sample_rate'],
                                'physical_max': ((204700 - int(in_file.header['y offset']))
                                                 / int(in_file.header['y gain'])),
                                'physical_min': ((-204800 - int(in_file.header['y offset']))
                                                 / int(in_file.header['y gain'])),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Accelerometer z",
                                'transducer': "MEMS Accelerometer",
                                'dimension': in_file.header['Accelerometer Units'],
                                'sample_rate': in_file.data['sample_rate'],
                                'physical_max': ((204700 - int(in_file.header['z offset']))
                                                 / int(in_file.header['z gain'])),
                                'physical_min': ((-204800 - int(in_file.header['z offset']))
                                                 / int(in_file.header['z gain'])),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Temperature",
                                'transducer': "Linear active thermistor",
                                'dimension': in_file.header['Temperature Sensor Units'],
                                'sample_rate': in_file.data['temperature_sample_rate'],
                                'physical_max': int(in_file.header["Temperature Sensor Range"][5:7]),
                                'physical_min': int(in_file.header["Temperature Sensor Range"][0]),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Light",
                                'transducer': "Silicon photodiode",
                                'dimension': in_file.header['Light Meter Units'],
                                'sample_rate': in_file.data['sample_rate'],
                                'physical_max': 1023 * int(in_file.header['Lux']) / int(in_file.header['Volts']),
                                'physical_min': 0 * int(in_file.header['Lux']) / int(in_file.header['Volts']),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Button",
                                'transducer': "Mechanical membrane switch",
                                'dimension': '',
                                'sample_rate': in_file.data['sample_rate'],
                                'physical_max': 1,
                                'physical_min': 0,
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''}]

        # set signal data and trim appropriately
        self.signals = [in_file.data['x'][trim_samples:],
                        in_file.data['y'][trim_samples:],
                        in_file.data['z'][trim_samples:],
                        in_file.data['temperature'],
                        in_file.data['light'][trim_samples:],
                        in_file.data['button'][trim_samples:]]

        return True

    def import_axivity(self, file_path, resample=True, quiet=False):
        """

        Args:
            file_path:
            resample:
            quiet:

        Returns:

        """

        file_path = Path(file_path)

        # check if file exists
        if not file_path.is_file():
            print("Import failed: file does not exist.")
            return False

        in_file = CWAFile(file_path)
        in_file.read(resample=resample, quiet=quiet)

        device_type = in_file.header['device_type']

        if device_type == 'AX6':
            device_type = 'AXV6'
        else:
            print(f'{device_type} not currently supported.')
            return False

        # find amount to trim from start to shift to next second integer (EDF files don't store start time to ms)
        trim_microseconds = (1000000 - in_file.header['start_time'].microsecond
                             if in_file.header['start_time'].microsecond > 0
                             else 0)
        trim_samples = round(in_file.header['sample_rate'] * (trim_microseconds / 1000000))
        trim_packets = round(in_file.header['packet_rate'] * (trim_microseconds / 1000000))

        meta_keys = in_file.header['metadata'].keys()

        subject_code = in_file.header['metadata']['subject_code'] if 'subject_code' in meta_keys else ''
        study_code = in_file.header['metadata']['study_code'] if 'study_code' in meta_keys else ''
        coll_id = in_file.header['metadata']['_sn'] if '_sn' in meta_keys else ''
        body_location = in_file.header['metadata']['body_location'].replace(' ', '_') \
            if 'body_location' in meta_keys else ''
        sex = in_file.header['metadata']['sex'] if 'sex' in meta_keys else ''

        # update header to match nwdata format
        # self.header = {'patientcode': subject_code,
        #                'gender': sex,
        #                'birthdate': '',
        #                'patientname': '',
        #                'patient_additional': coll_id,
        #                'startdate': in_file.header['start_time'] + dt.timedelta(microseconds=trim_microseconds),
        #                'admincode': study_code,
        #                'technician': '',
        #                'equipment': device_type + '_' + str(in_file.header['device_id']),
        #                'recording_additional': body_location}

        self.header = {'study_code': study_code,
                       'subject_id': subject_code,
                       'coll_id': coll_id,
                       'name': '',
                       'sex': sex,
                       'birthdate': '',
                       'patient_additional': '',
                       'start_datetime': in_file.header['start_time'] + dt.timedelta(microseconds=trim_microseconds),
                       'config_datetime': in_file.header['last_change_time'],
                       'technician': '',
                       'device_type': device_type,
                       'device_id': str(in_file.header['device_id']),
                       'device_location': body_location,
                       'recording_additional': ''}

        # update signal headers to match nwdata
        new_signal_headers = {'gx': {'label': "Gyroscope x",
                                     'transducer': "MEMS",
                                     'dimension': 'degree/s',
                                     'sample_rate': in_file.header['sample_rate'],
                                     'physical_max': max(in_file.data['gx']),
                                     'physical_min': min(in_file.data['gx']),
                                     'digital_max': 32767,
                                     'digital_min': -32768,
                                     'prefilter': ''},
                              'gy': {'label': "Gyroscope y",
                                     'transducer': "MEMS",
                                     'dimension': 'degree/s',
                                     'sample_rate': in_file.header['sample_rate'],
                                     'physical_max': max(in_file.data['gy']),
                                     'physical_min': min(in_file.data['gy']),
                                     'digital_max': 32767,
                                     'digital_min': -32768,
                                     'prefilter': ''},
                              'gz': {'label': "Gyroscope z",
                                     'transducer': "MEMS",
                                     'dimension': 'degree/s',
                                     'sample_rate': in_file.header['sample_rate'],
                                     'physical_max': max(in_file.data['gz']),
                                     'physical_min': min(in_file.data['gz']),
                                     'digital_max': 32767,
                                     'digital_min': -32768,
                                     'prefilter': ''},
                              'ax': {'label': "Accelerometer x",
                                     'transducer': "MEMS",
                                     'dimension': 'g',
                                     'sample_rate': in_file.header['sample_rate'],
                                     'physical_max': max(in_file.data['ax']),
                                     'physical_min': min(in_file.data['ax']),
                                     'digital_max': 32767,
                                     'digital_min': -32768,
                                     'prefilter': ''},
                              'ay': {'label': "Accelerometer y",
                                     'transducer': "MEMS",
                                     'dimension': 'g',
                                     'sample_rate': in_file.header['sample_rate'],
                                     'physical_max': max(in_file.data['ay']),
                                     'physical_min': min(in_file.data['ay']),
                                     'digital_max': 32767,
                                     'digital_min': -32768,
                                     'prefilter': ''},
                              'az': {'label': "Accelerometer z",
                                     'transducer': "MEMS",
                                     'dimension': 'g',
                                     'sample_rate': in_file.header['sample_rate'],
                                     'physical_max': max(in_file.data['az']),
                                     'physical_min': min(in_file.data['az']),
                                     'digital_max': 32767,
                                     'digital_min': -32768,
                                     'prefilter': ''},
                              'light': {'label': "Light",
                                        'transducer': "Logarithmic light sensor",
                                        'dimension': 'units',
                                        'sample_rate': in_file.header['packet_rate'],
                                        'physical_max': max(in_file.data['light']),
                                        'physical_min': min(in_file.data['light']),
                                        'digital_max': 32767,
                                        'digital_min': -32768,
                                        'prefilter': ''},
                              'temperature': {'label': "Temperature",
                                              'transducer': "Linear thermistor",
                                              'dimension': 'C',
                                              'sample_rate': in_file.header['packet_rate'],
                                              'physical_max': max(in_file.data['temperature']),
                                              'physical_min': min(in_file.data['temperature']),
                                              'digital_max': 32767,
                                              'digital_min': -32768,
                                              'prefilter': ''}
                              }

        self.signal_headers = []
        self.signals = []

        for key in in_file.data:

            if key in new_signal_headers.keys():
                self.signal_headers.append(new_signal_headers[key])

                if key in ['gx', 'gy', 'gz', 'ax', 'ay', 'az']:
                    self.signals.append(in_file.data[key][trim_samples:])
                elif key in ['light', 'temperature']:
                    self.signals.append(in_file.data[key][trim_packets:])

        return True

    def import_nonin(self, file_path, quiet=False):
        """Imports data from a Nonin-device-generated ASCII file.

        Args:
            file_path (str): Full absolute path to the file to be imported
            quiet (bool): Suppress messages, default is False
        Returns:
            True if successful, False otherwise.
        """

        file_path = Path(file_path)

        # check if file exists
        if not file_path.is_file():
            print("Import failed: file does not exist.")
            return False

        # read file
        in_file = NoninFile(file_path)
        in_file.read(quiet=quiet)

        if in_file.header['gender'] == 1:
            gender = 1
        elif in_file.header['gender'] == 2:
            gender = 0
        else:
            gender = 2

        # update header to match nwdata
        self.header = {'patientcode': in_file.header['id'],
                       'gender': gender,
                       'birthdate': in_file.header['dob'],
                       'patientname': in_file.header['first_name'] + '_' + in_file.header['last_name'],
                       'patient_additional': '',
                       'startdate': dt.datetime.combine(in_file.header['start_date'], in_file.header['start_time']),
                       'admincode': '',
                       'technician': in_file.header['physician'],
                       'equipment': 'NOWO',
                       'recording_additional': ''}

        # update signal headers to match nwdata
        self.signal_headers = [{'label': "Pulse",
                                'transducer': "Pulse oximeter",
                                'dimension': "bpm",
                                'sample_rate': in_file.header['sample_rate'],
                                'physical_max': max(in_file.data['pulse']),
                                'physical_min': (min(in_file.data['pulse'])
                                                 if max(in_file.data['pulse']) > min(in_file.data['pulse'])
                                                 else max(in_file.data['pulse']) - 1),
                                'digital_max': max(in_file.data['pulse']),
                                'digital_min': (min(in_file.data['pulse'])
                                                if max(in_file.data['pulse']) > min(in_file.data['pulse'])
                                                else max(in_file.data['pulse']) - 1),
                                'prefilter': ''},
                               {'label': "SpO2",
                                'transducer': "Pulse oximeter",
                                'dimension': '%',
                                'sample_rate': in_file.header['sample_rate'],
                                'physical_max': max(in_file.data['spo2']),
                                'physical_min': (min(in_file.data['spo2'])
                                                 if max(in_file.data['spo2']) > min(in_file.data['spo2'])
                                                 else max(in_file.data['spo2']) - 1),
                                'digital_max': max(in_file.data['spo2']),
                                'digital_min': (min(in_file.data['spo2'])
                                                if max(in_file.data['spo2']) > min(in_file.data['spo2'])
                                                else max(in_file.data['spo2']) - 1),
                                'prefilter': ''}]

        # set signal data
        self.signals = [in_file.data['pulse'],
                        in_file.data['spo2']]

        return True
