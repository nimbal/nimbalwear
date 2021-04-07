import datetime as dt
import copy

from nwfiles import GENEActiv as gnac
from nwfiles import EDF as edf
from nwfiles import Nonin as nonw

class nwdata:

    def __init__(self):

        self.header = {'patientcode': '',
                       'gender': 2,
                       'birthdate': '',
                       'patientname': '',
                       'patient_additional': '',
                       'startdate': '',
                       'admincode': '',
                       'technician': '',
                       'equipment': '',
                       'recording_additional': ''}

        self.signal_headers = []
        self.signals = []

    def crop(self, new_start_time, new_end_time):

        start_time = self.header['startdate']
        duration = dt.timedelta(seconds=len(self.signals[0]) * self.signal_headers[0]['sample_rate'])
        end_time = start_time + duration

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

        for sig_num in range(len(self.signal_headers)):

            # calculate start and duration samples
            crop_start = int((new_start_time - start_time).total_seconds()
                             * self.signal_headers[sig_num]['sample_rate'])
            crop_duration = int((new_end_time - new_start_time).total_seconds()
                                * self.signal_headers[sig_num]['sample_rate'])
            crop_end = crop_start + crop_duration

            self.signals[sig_num] = self.signals[sig_num][crop_start:crop_end]

    def deidentify(self):

        self.header.update({'gender': 2, 'birthdate': '', 'patientname': ''})

    def export_edf(self, file_path='', sig_nums_out=None):

        signals = copy.deepcopy(self.signals)
        signal_headers = copy.deepcopy(self.signal_headers)

        if sig_nums_out is not None:
            signals = [signals[sig_num] for sig_num in sig_nums_out]
            signal_headers = [signal_headers[sig_num] for sig_num in sig_nums_out]

        edf.write(self.header, signal_headers, signals, file_path)

    def import_edf(self, file_path):

        self.header, self.signal_headers, self.signals = edf.read(file_path)

    def import_gnac(self, file_path, parse_data=True, start=1, end=-1, downsample=1,
                    calibrate=True, correct_drift=False, update=True, quiet=False):

        ga_file = gnac.GENEActivFile(file_path)
        ga_file.read(parse_data=parse_data, start=start, end=end, downsample=downsample,
                     calibrate=calibrate, correct_drift=correct_drift, update=update, quiet=quiet)

        trim_microseconds = (1000000 - ga_file.data['start_time'].microsecond
                             if ga_file.data['start_time'].microsecond > 0
                             else 0)

        trim_samples = round(ga_file.data['sample_rate'] * (trim_microseconds / 1000000))

        self.header = {'patientcode': ga_file.header['Subject Code'],
                       'gender': ga_file.header['Sex'],
                       'birthdate': dt.datetime.strptime(ga_file.header["Date of Birth"], "%Y-%m-%d"),
                       'patientname': '',
                       'patient_additional': '',
                       'startdate': ga_file.data['start_time'] + dt.timedelta(microseconds=trim_microseconds),
                       'admincode': ga_file.header['Study Code'],
                       'technician': ga_file.header['Investigator ID'],
                       'equipment': ga_file.header['Device Type'] + '_' + ga_file.header['Device Unique Serial Code'],
                       'recording_additional': ga_file.header['Device Location Code'].replace(' ', '_')}

        self.signal_headers = [{'label': "Accelerometer x",
                                'transducer': "MEMS Accelerometer",
                                'dimension': ga_file.header['Accelerometer Units'],
                                'sample_rate': ga_file.data['sample_rate'],
                                'physical_max': (204700 - int(ga_file.header['x offset'])) / int(ga_file.header['x gain']),
                                'physical_min': (-204800 - int(ga_file.header['x offset'])) / int(ga_file.header['x gain']),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Accelerometer y",
                                'transducer': "MEMS Accelerometer",
                                'dimension': ga_file.header['Accelerometer Units'],
                                'sample_rate': ga_file.data['sample_rate'],
                                'physical_max': (204700 - int(ga_file.header['y offset'])) / int(ga_file.header['y gain']),
                                'physical_min': (-204800 - int(ga_file.header['y offset'])) / int(ga_file.header['y gain']),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Accelerometer z",
                                'transducer': "MEMS Accelerometer",
                                'dimension': ga_file.header['Accelerometer Units'],
                                'sample_rate': ga_file.data['sample_rate'],
                                'physical_max': (204700 - int(ga_file.header['z offset'])) / int(ga_file.header['z gain']),
                                'physical_min': (-204800 - int(ga_file.header['z offset'])) / int(ga_file.header['z gain']),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Temperature",
                                'transducer': "Linear active thermistor",
                                'dimension': ga_file.header['Temperature Sensor Units'],
                                'sample_rate': ga_file.data['temperature_sample_rate'],
                                'physical_max': int(ga_file.header["Temperature Sensor Range"][5:7]),
                                'physical_min': int(ga_file.header["Temperature Sensor Range"][0]),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Light",
                                'transducer': "Silicon photodiode",
                                'dimension': ga_file.header['Light Meter Units'],
                                'sample_rate': ga_file.data['sample_rate'],
                                'physical_max': 1023 * int(ga_file.header['Lux']) / int(ga_file.header['Volts']),
                                'physical_min': 0 * int(ga_file.header['Lux']) / int(ga_file.header['Volts']),
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''},
                               {'label': "Button",
                                'transducer': "Mechanical membrane switch",
                                'dimension': '',
                                'sample_rate': ga_file.data['sample_rate'],
                                'physical_max': 1,
                                'physical_min': 0,
                                'digital_max': 32767,
                                'digital_min': -32768,
                                'prefilter': ''}]

        self.signals = [ga_file.data['x'][trim_samples:],
                        ga_file.data['y'][trim_samples:],
                        ga_file.data['z'][trim_samples:],
                        ga_file.data['temperature'],
                        ga_file.data['light'][trim_samples:],
                        ga_file.data['button'][trim_samples:]]

    def import_nonw(self, file_path, quiet=False):

        nonw_file = nonw.NoninFile(file_path)
        nonw_file.read(quiet=quiet)

        self.header = {'patientcode': nonw_file.header['id'],
                       'gender': nonw_file.header['gender'],
                       'birthdate': nonw_file.header['dob'],
                       'patientname': nonw_file.header['first_name'] + '_' + nonw_file.header['last_name'],
                       'patient_additional': '',
                       'startdate': dt.datetime.combine(nonw_file.header['start_date'], nonw_file.header['start_time']),
                       'admincode': '',
                       'technician': nonw_file.header['physician'],
                       'equipment': 'Nonin',
                       'recording_additional': ''}

        self.signal_headers = [{'label': "Pulse",
                                'transducer': "Pulse oximeter",
                                'dimension': "bpm",
                                'sample_rate': nonw_file.header['sample_rate'],
                                'physical_max': max(nonw_file.data['pulse']),
                                'physical_min': (min(nonw_file.data['pulse'])
                                                 if max(nonw_file.data['pulse']) > min(nonw_file.data['pulse'])
                                                 else max(nonw_file.data['pulse']) - 1),
                                'digital_max': max(nonw_file.data['pulse']),
                                'digital_min': (min(nonw_file.data['pulse'])
                                                if max(nonw_file.data['pulse']) > min(nonw_file.data['pulse'])
                                                else max(nonw_file.data['pulse']) - 1),
                                'prefilter': ''},
                               {'label': "SpO2",
                                'transducer': "Pulse oximeter",
                                'dimension': '%',
                                'sample_rate': nonw_file.header['sample_rate'],
                                'physical_max': max(nonw_file.data['spo2']),
                                'physical_min': (min(nonw_file.data['spo2'])
                                                if max(nonw_file.data['spo2']) > min(nonw_file.data['spo2'])
                                                else max(nonw_file.data['spo2']) - 1),
                                'digital_max': max(nonw_file.data['spo2']),
                                'digital_min': (min(nonw_file.data['spo2'])
                                               if max(nonw_file.data['spo2']) > min(nonw_file.data['spo2'])
                                               else max(nonw_file.data['spo2']) - 1),
                                'prefilter': ''}]

        self.signals = [nonw_file.data['pulse'],
                        nonw_file.data['spo2']]
