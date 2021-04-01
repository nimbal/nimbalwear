import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
from fnmatch import fnmatch
from tqdm import tqdm
import nwfiles.file.EDF as edf

file_patterns = ['*']

device_path = '/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/cropped_sensor_edf'

sensor_paths = ['ACCELEROMETER',
                'TEMPERATURE',
                'LIGHT',
                'BUTTON']

for sensor_path in tqdm(sensor_paths):

    edf_path = os.path.join(device_path, sensor_path)

    csv_path = os.path.join(edf_path, 'edf-headers.csv')

    file_list = [f for f in os.listdir(edf_path)
                 if f.lower().endswith('.edf') and not f.startswith('.')
                 and any([fnmatch(f, file_pattern) for file_pattern in file_patterns])]
    file_list.sort()

    for file_name in tqdm(file_list):

        file_base = os.path.splitext(file_name)[0]
        file_base_parts = file_base.split('_')

        with open(os.path.join(edf_path, file_name), 'rb') as edf_file:
            edf_file.seek(8, 0)
            patient_id = [item for item in edf_file.read(80).decode().split(' ') if item is not '']
            recording_id = [item for item in edf_file.read(80).decode().split(' ') if item is not '']

        if len(patient_id) < 5:
            patient_id += [''] * (5 - len(patient_id))

        if len(recording_id) < 6:
            recording_id += [''] * (6 - len(recording_id))

        patient_id[0] = file_base_parts[2]
        patient_id[4] = '_'.join(['VISIT', file_base_parts[3]])
        recording_id[2] = file_base_parts[0]
        recording_id[5] = file_base_parts[5]

        with open(os.path.join(edf_path, file_name), 'r+b') as edf_file:
            edf_file.seek(8, 0)
            edf_file.write('{:80.80}'.format(' '.join(patient_id)).encode())
            edf_file.write('{:80.80}'.format(' '.join(recording_id)).encode())

    edf.header_summary(edf_path, csv_path, quiet=True)