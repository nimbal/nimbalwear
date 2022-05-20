import textract
import os
import csv
import time
import datetime

from nwdata.nwfiles import EDF

class NoninFile:

    def __init__(self, file_path):

        self.file_path = os.path.abspath(file_path)
        self.file_name = os.path.basename(self.file_path)
        self.file_dir = os.path.dirname(self.file_path)

        self.header = {
            'version': None,
            'id': None,
            'first_name': None,
            'last_name': None,
            'physician': None,
            'height': None,
            'weight': None,
            'dob': '',
            'age': None,
            'gender': None,
            'sample_rate': None,
            'start_date': None,
            'start_time': None}

        self.data = {
            'pulse': [],
            'spo2': []}

    def read(self, quiet=False):

        read_start_time = time.time()

        # if file does not exist then exit
        if not os.path.exists(self.file_path):
            print(f"****** WARNING: {self.file_path} does not exist.\n")
            return

        # Read Nonin .asc file
        if not quiet:
            print("Reading %s ..." % self.file_path)

        with open(self.file_path, 'r') as asc_file:
            lines = [line[:-1].strip('"') for line in asc_file.readlines()]

        # parse header
        self.header.update({
            'version': lines[0].split('=')[1].split(',')[0],
            'id': lines[7],
            'first_name': lines[1],
            'last_name': lines[2],
            'physician': lines[4],
            'height': float(lines[5]),
            'weight': float(lines[6]),
            'dob': (datetime.datetime.strptime(lines[8].replace('-', '/'), '%Y/%m/%d').date()
                    if not lines[8] == ''
                    else ''),
            'age': int(lines[9]),  # not certain about this field
            'gender': lines[10]})  # 2 = Female

        # check version - only tested on version 3
        if not self.header['version'] == '3':
            print(f"****** WARNING: Version not equal to 3. Data may be unreliable..\n")

        # get dates and calculate sample rate
        first_sample = lines[17].split(',')
        first_sample = [int(s) for s in first_sample]
        first_sample_time = datetime.datetime(first_sample[1], first_sample[2], first_sample[3], first_sample[4],
                                              first_sample[5], first_sample[6])

        second_sample = lines[18].split(',')
        second_sample = [int(s) for s in second_sample]
        second_sample_time = datetime.datetime(second_sample[1], second_sample[2], second_sample[3], second_sample[4],
                                               second_sample[5], second_sample[6])

        last_sample = lines[-1].split(',')
        last_sample = [int(s) for s in last_sample]
        last_sample_time = datetime.datetime(last_sample[1], last_sample[2], last_sample[3], last_sample[4],
                                             last_sample[5], last_sample[6])

        self.header.update({
            'sample_rate': 1 / (second_sample_time - first_sample_time).seconds,
            'start_date': datetime.date(first_sample[1], first_sample[2], first_sample[3]),
            'start_time': datetime.time(first_sample[4], first_sample[5], first_sample[6])})

        # check that sample count matches start and end datetime
        if not len(lines) - 18 == (last_sample_time - first_sample_time).total_seconds() * self.header['sample_rate']:
            print(f"****** WARNING: MISSING DATA. SAMPLE NUMBER DOES NOT MATCH TIME.\n")

        # parse data
        self.data.update({'pulse': [int(line.split(',')[7]) for line in lines[17:]],
                          'spo2': [int(line.split(',')[8]) for line in lines[17:]]})

        if not quiet:
            print("Done reading file. Time to read file: ", time.time() - read_start_time, "seconds.")

    def write(self, file_type='edf', out_file='', edf_header={}, edf_signal_headers=[], deid=False, quiet=False):

        # check whether data has been read

        error_str = "Write failed: "

        # check that header, data exist and are same length, etc.
        if not len(self.data['pulse']):
            print(error_str + "No data.")
            return

        if not isinstance(self.header['start_date'], datetime.date):
            print(error_str + "start_date is not a datetime object - header may not have been imported correctly.")
            return

        if not isinstance(self.header['start_time'], datetime.time):
            print(error_str + "start_time is not a datetime object - header may not have been imported correctly.")
            return

        if file_type == 'edf':

            # TODO: CHECK THAT FILE EXTENSION MATCHES file_type??

            if out_file == '':
                out_file = os.path.join(self.file_dir, self.file_name[:-3] + 'edf')

            # TODO: check whether path exists

            if not quiet:
                print("Writing %s ..." % out_file)

            if self.header['gender'] == 1:
                gender = 1
            elif self.header['gender'] == 2:
                gender = 0
            else:
                gender = ''

            header = {'patientcode': self.header['id'],
                      'gender': gender,
                      'birthdate': self.header['dob'],
                      'patientname': self.header['first_name'] + '_' + self.header['last_name'],
                      'patient_additional': '',
                      'startdate': datetime.datetime.combine(self.header['start_date'], self.header['start_time']),
                      'admincode': '',
                      'technician': self.header['physician'],
                      'equipment': 'Nonin',
                      'recording_additional': ''}

            if deid:
                header.update({'gender': 2,
                               'birthdate': '',
                               'patientname': ''})

            header.update(edf_header)

            signal_headers = [{'label': "Pulse",
                               'transducer': "Pulse oximeter",
                               'dimension': "bpm",
                               'sample_rate': self.header['sample_rate'],
                               'physical_max': max(self.data['pulse']),
                               'physical_min': (min(self.data['pulse'])
                                                if max(self.data['pulse']) > min(self.data['pulse'])
                                                else max(self.data['pulse']) - 1),
                               'digital_max': max(self.data['pulse']),
                               'digital_min': (min(self.data['pulse'])
                                               if max(self.data['pulse']) > min(self.data['pulse'])
                                               else max(self.data['pulse']) - 1),
                               'prefilter': ''},
                              {'label': "SpO2",
                               'transducer': "Pulse oximeter",
                               'dimension': '%',
                               'sample_rate': self.header['sample_rate'],
                               'physical_max': max(self.data['spo2']),
                               'physical_min': (min(self.data['spo2'])
                                                if max(self.data['spo2']) > min(self.data['spo2'])
                                                else max(self.data['spo2']) - 1),
                               'digital_max': max(self.data['spo2']),
                               'digital_min': (min(self.data['spo2'])
                                               if max(self.data['spo2']) > min(self.data['spo2'])
                                               else max(self.data['spo2']) - 1),
                               'prefilter': ''}]

            sh_index = 0

            for signal_header in edf_signal_headers:
                signal_headers[sh_index].update(signal_header)
                sh_index = sh_index + 1

            # write to edf
            edf_file = EDF.EDFFile(out_file)
            edf_file.header = header
            edf_file.signal_headers = signal_headers
            if len(self.data['pulse']) > 0:
                edf_file.signals = [self.data['pulse'], self.data['spo2']]
            edf_file.write(out_file, quiet=quiet)


def convert_nonw_dir_edf(in_path, out_path, deid=False):

    file_list = [f for f in os.listdir(in_path) if f.lower().endswith('.asc')]
    file_list.sort()

    for file_name in file_list:
        nonw_file = NoninFile(os.path.join(in_path, file_name))
        nonw_file.read()
        nonw_file.write(out_file=os.path.join(out_path, file_name[:-3] + 'edf'), deid=deid)


def scrape_nonw_pdf(file_path):

    # scrape pdf for text
    text = textract.process(file_path)
    text = text.decode('utf-8')

    # create list of text 'lines'
    lines = text.splitlines()

    # find reference label positions (using this instead of .index() to account for random spaces scraped from pdf
    name_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == 'Name:'][0]
    height_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == 'Height:'][0]
    rec_date_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == 'RecordingDate:'][0]
    spo2_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == 'SpO2'][0]
    pulse_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == 'Pulse'][0]
    spo2_level_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == '%SpO2LevelEvents'][0]
    spo2_below_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == 'Below(%)Time(%)'][0]
    parameter_index = [index for index, value in enumerate(lines) if value.replace(' ', '') == 'AnalysisParameters'][0]

    # check if fields were enetered
    is_physician = (name_index == 10)
    is_dob = (height_index == name_index + 7)
    spo2_88_shift = 2 if lines[spo2_below_index + 19].strip() == '***' else 0

    # create dictionary of data scraped from pdf
    data = {'id': lines[height_index + 13],
            'name': lines[name_index + 3],
            'gender': lines[height_index + 7],
            'physician': lines[6] if is_physician else '',
            'age': lines[name_index - 2],
            'dob': lines[name_index + 5] if is_dob else '',
            'height': lines[height_index + 4],
            'weight': lines[height_index + 5],
            'bmi': lines[height_index + 11],
            'note_1': lines[name_index - 1],
            'note_2': lines[height_index + 6],
            'recording_date': lines[rec_date_index + 4],
            'start_time': lines[rec_date_index + 8],
            'duration': lines[rec_date_index + 12],
            'analyzed': lines[rec_date_index + 16],
            'comments': lines[rec_date_index + 2],
            'spo2_events': lines[spo2_index + 1],
            'spo2_events_time': lines[spo2_index + 2],
            'spo2_events_time_avg': lines[spo2_index + 3],
            'spo2_desat_index': lines[spo2_index + 5],
            'spo2_artifact': lines[spo2_index + 6],
            'spo2_desat_index_adj': lines[spo2_index + 7],
            'pulse_events': lines[pulse_index + 1],
            'pulse_events_time': lines[pulse_index + 2],
            'pulse_events_time_avg': lines[pulse_index + 3],
            'pulse_index': lines[pulse_index + 4],
            'pulse_artifact': lines[pulse_index + 6],
            'pulse_index_adj': lines[pulse_index + 7],
            'spo2_99to95_events': lines[spo2_level_index + 10],
            'spo2_94to90_events': lines[spo2_level_index + 11],
            'spo2_89to85_events': lines[spo2_level_index + 12],
            'spo2_84to80_events': lines[spo2_level_index + 13],
            'spo2_79to75_events': lines[spo2_level_index + 14],
            'spo2_74to70_events': lines[spo2_level_index + 15],
            'spo2_69to65_events': lines[spo2_level_index + 16],
            'spo2_64to60_events': lines[spo2_level_index + 17],
            'spo2_below_100': lines[spo2_below_index + 10],
            'spo2_below_95': lines[spo2_below_index + 11],
            'spo2_below_90': lines[spo2_below_index + 12],
            'spo2_below_85': lines[spo2_below_index + 13],
            'spo2_below_80': lines[spo2_below_index + 14],
            'spo2_below_75': lines[spo2_below_index + 15],
            'spo2_below_70': lines[spo2_below_index + 16],
            'spo2_below_65': lines[spo2_below_index + 17],
            'spo2_basal': lines[spo2_below_index + 19 + spo2_88_shift],
            'spo2_time_below_88': lines[spo2_below_index + 20 + spo2_88_shift],
            'spo2_events_below_88': lines[spo2_below_index + 21 + spo2_88_shift],
            'spo2_max_duration_below_88':
                lines[spo2_below_index + 23 - (spo2_88_shift * 2)].split('sec')[0]
                if not lines[spo2_below_index + 23 - (spo2_88_shift * 2)].strip() == '***' else '',
            'spo2_max_duration_below_88_time':
                lines[spo2_below_index + 23 - (spo2_88_shift * 2)].split('at')[1]
                if not lines[spo2_below_index + 23 - (spo2_88_shift * 2)].strip() == '***' else '',
            'spo2_min':
                lines[spo2_below_index + 25].split('at')[0]
                if not lines[spo2_below_index + 25].strip() == '***' else '',
            'spo2_min_time':
                lines[spo2_below_index + 25].split('at')[1]
                if not lines[spo2_below_index + 25].strip() == '***' else '',
            'spo2_max':
                lines[spo2_below_index + 26].split('at')[0]
                if not lines[spo2_below_index + 26].strip() == '***' else '',
            'spo2_max_time':
                lines[spo2_below_index + 26].split('at')[1]
                if not lines[spo2_below_index + 26].strip() == '***' else '',
            'spo2_avg_low':
                lines[spo2_below_index + 27]
                if not lines[spo2_below_index + 27].strip() == '***' else '',
            'spo2_avg_low_below_88':
                lines[spo2_below_index + 28]
                if not lines[spo2_below_index + 28].strip() == '***' else '',
            'pulse_avg': lines[spo2_below_index + 30],
            'pulse_min': lines[spo2_below_index + 31],
            'pulse_max': lines[spo2_below_index + 32],
            'spo2_param_change':
                lines[parameter_index + 1].replace(' ', '').split('atleast')[1].split('%')[0],
            'spo2_param_duration':
                lines[parameter_index + 1].replace(' ', '').split('durationof')[1].split('seconds')[0],
            'pulse_param_change':
                lines[parameter_index + 2].replace(' ', '').split('atleast')[1].split('bpm')[0],
            'pulse_param_duration':
                lines[parameter_index + 2].replace(' ', '').split('durationof')[1].split('seconds')[0]}

    # strip whitespace from all values in data dict
    return {k: v.strip() for k, v in data.items()}


def scrape_nonw_pdf_dir(dir_path, write_csv=False):

    file_list = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
    file_list.sort()

    dicts = []

    for file_name in file_list:
        print('Scraping', file_name)
        dicts.append(scrape_nonw_pdf(os.path.join(dir_path, file_name)))

    data = {}

    for key in dicts[0].keys():
        data[key] = [d[key] for d in dicts]

    data_list = [data[key] for key in data.keys()]

    data_list = [*zip(*data_list)]

    if write_csv:
        csv_path = os.path.join(dir_path, 'OND06_01_ALL_SNSR_NONW.csv')
        print('Writing', csv_path)
        with open(csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data.keys())
            writer.writerows(data_list)

    return data

#
