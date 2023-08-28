import os
import csv
import copy
import math
import datetime as dt
from pathlib import Path

import pyedflib
from tqdm import tqdm
import numpy as np


class EDFFile:

    def __init__(self, file_path):

        # initialize attributes
        self.file_path = Path(file_path).absolute()
        self.file_name = self.file_path.name
        self.file_dir = self.file_path.parent

        self.header = {}
        self.signal_headers = []
        self.signals = []

    def read_header(self):

        # read header only - does not use pyedflib, reads as binary

        # check if file exists
        if not self.file_path.is_file():
            print("Read header failed: file does not exist.")
            return

        signal_headers = []

        # read binary header and signal headers into variables
        with open(self.file_path, 'rb') as edf_file:
            edf_file.seek(8, 0)
            patient_id = [item for item in edf_file.read(80).decode().split(' ') if item != '']
            recording_id = [item for item in edf_file.read(80).decode().split(' ') if item != '']
            edf_file.seek(8, 1)
            start_time = edf_file.read(8).decode()
            edf_file.seek(52, 1)
            dr_count = int(edf_file.read(8).decode().strip())
            dr_duration = int(edf_file.read(8).decode().strip())
            ns = int(edf_file.read(4).decode().strip())
            label = [edf_file.read(16).decode().strip() for sig in range(ns)]
            transducer = [edf_file.read(80).decode().strip() for sig in range(ns)]
            dimension = [edf_file.read(8).decode().strip() for sig in range(ns)]
            physical_min = [edf_file.read(8).decode().strip() for sig in range(ns)]
            physical_max = [edf_file.read(8).decode().strip() for sig in range(ns)]
            digital_min = [edf_file.read(8).decode().strip() for sig in range(ns)]
            digital_max = [edf_file.read(8).decode().strip() for sig in range(ns)]
            prefilter = [edf_file.read(80).decode().strip() for sig in range(ns)]
            samples_dr = [int(edf_file.read(8).decode().strip()) for sig in range(ns)]

        startdate = dt.datetime.strptime(' '.join([recording_id[1], start_time]), '%d-%b-%Y %H.%M.%S')

        birthdate = '' if patient_id[2] == 'X' else dt.datetime.strptime(patient_id[2], '%d-%b-%Y')

        # transfer appropriate variables into header dict
        header = {'patientcode': patient_id[0],
                  'sex': patient_id[1],
                  'birthdate': birthdate,
                  'patientname': patient_id[3],
                  'patient_additional': ' '.join(patient_id[4:]) if len(patient_id) > 4 else '',
                  'startdate': startdate,
                  'duration': dt.timedelta(seconds=(dr_count * dr_duration)),
                  'admincode': recording_id[2],
                  'technician': recording_id[3],
                  'equipment': recording_id[4],
                  'recording_additional': ' '.join(recording_id[5:]) if len(recording_id) > 5 else ''}

        # transfer appropriate variables into list of signal_header dicts
        for sig in range(ns):

            if label[sig] != 'EDF Annotations':

                signal_headers.append({'label': label[sig],
                                       'transducer': transducer[sig],
                                       'dimension': dimension[sig],
                                       'sample_dr': samples_dr[sig],
                                       'sample_rate': samples_dr[sig] / dr_duration,
                                       'physical_max': float(physical_max[sig]),
                                       'physical_min': float(physical_min[sig]),
                                       'digital_max': int(digital_max[sig]),
                                       'digital_min': int(digital_min[sig]),
                                       'prefilter': prefilter[sig]})

        # set object attributes
        self.header = header
        self.signal_headers = signal_headers

    def read(self, quiet=False):

        # read file

        # check if file exists
        if not self.file_path.exists:
            print("Read failed: file does not exist.")
            return

        # Read EDf  file
        if not quiet:
            print("Reading %s ..." % self.file_path)

        # read headers with custom method
        self.read_header()

        # read edf file signals
        edf_reader = pyedflib.EdfReader(str(self.file_path))
        self.signals = [edf_reader.readSignal(sig_num) for sig_num in range(edf_reader.signals_in_file)]
        edf_reader.close()

    def write(self, out_file, sig_nums_out=None, quiet=False):

        out_file = Path(out_file)

        # error handling
        error_str = "Write failed: "

        # ensure data in signals
        if not self.signals:
            print(error_str + "no signal data.")
            return

        # ensure signal_header and signals length match
        if not len(self.signal_headers) == len(self.signals):
            print(error_str + "number of signal headers does not match number of signals.")
            return

        # ensure header exists
        if not self.header:
            print(error_str + "no header data.")
            return

        # ensure valid sig nums used as arguments
        if sig_nums_out is not None:
            if any([sig_num >= len(self.signals) for sig_num in sig_nums_out]):
                print(error_str + "invalid signal number.")
                return

        if not quiet:
            print("Writing %s ..." % out_file)

        # check that all folders exist for data output file
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # create deep copy of mutable objects
        signals = copy.deepcopy(self.signals)
        signal_headers = copy.deepcopy(self.signal_headers)

        # get specified signals to export
        if sig_nums_out is not None:
            signals = [signals[sig_num] for sig_num in sig_nums_out]
            signal_headers = [signal_headers[sig_num] for sig_num in sig_nums_out]

        # replace sample_rate key with sample_frequency for all signal headers
        for sig in range(len(signal_headers)):
            signal_headers[sig]['sample_frequency'] = signal_headers[sig].pop('sample_rate')

        # write edf
        edf_writer = pyedflib.EdfWriter(str(out_file), len(signals))
        edf_writer.setHeader(self.header)
        edf_writer.setSignalHeaders(signal_headers)
        edf_writer.writeSamples(signals)
        edf_writer.close()

    def crop(self, new_start_time=None, new_end_time=None):

        # crop data

        # check that header, signal_headers, signals exist and are same length, etc.
        error_str = "Crop failed: "

        if not self.signals:
            print(error_str + "No signal data.")
            return

        if not len(self.signal_headers) == len(self.signals):
            print(error_str + "number of signal headers does not match number of signals.")
            return

        if not self.header:
            print(error_str + "no header data.")
            return

        # read or calculate collection time info
        start_time = self.header['startdate']
        duration = dt.timedelta(seconds=len(self.signals[0]) * self.signal_headers[0]['sample_rate'])
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

        # loop through signals
        for sig_num in range(len(self.signal_headers)):

            # calculate new start, duration, and end samples
            crop_start = int((new_start_time - start_time).total_seconds()
                             * self.signal_headers[sig_num]['sample_rate'])
            crop_duration = int((new_end_time - new_start_time).total_seconds()
                                * self.signal_headers[sig_num]['sample_rate'])
            crop_end = crop_start + crop_duration

            # crop signal
            self.signals[sig_num] = self.signals[sig_num][crop_start:crop_end]


def edf_header_summary(edf_path='', csv_path='', quiet=False):

    # make work for folder or single file

    file_list = [f for f in os.listdir(edf_path)
                 if f.lower().endswith('.edf') and not f.startswith('.')]
    file_list.sort()

    header_dicts = []
    sig_header_dicts = []

    for file_name in tqdm(file_list):

        edf_file = EDFFile(os.path.join(edf_path, file_name))
        edf_file.read_header()

        header_dicts.append(edf_file.header)
        sig_header_dicts.append(edf_file.signal_headers)

    header = {}

    header_key_order = ['patientcode', 'sex', 'birthdate', 'patientname', 'patient_additional', 'startdate',
                        'duration', 'admincode', 'technician', 'equipment', 'recording_additional']

    for key in header_key_order:
        header[key] = [d[key] for d in header_dicts]

    sig_header = {}

    sig_header_key_order = ['label', 'transducer', 'dimension', 'sample_dr', 'sample_rate', 'physical_max',
                            'physical_min', 'digital_max', 'digital_min', 'prefilter']

    for signal in range(0, len(sig_header_dicts[0])):
        for key in sig_header_key_order:
            sig_header[key + '_' + str(signal)] = [d[signal][key] for d in sig_header_dicts]

    header.update(sig_header)

    header_list = [header[key] for key in header.keys()]

    header_list = [*zip(*header_list)]

    if not csv_path == '':
        if not quiet:
            print('Writing %s ...' % csv_path)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header.keys())
            writer.writerows(header_list)

    return header
