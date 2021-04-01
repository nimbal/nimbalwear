import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
import pyedflib
from tqdm import tqdm
import nwfiles.file.EDF as edf

device_path = '/Volumes/KIT_DATA/ReMiNDD/test_data/GENEActiv/sensor_edf'

sensor_paths = ['ACCELEROMETER',
                'TEMPERATURE',
                'LIGHT',
                'BUTTON']

for sensor_path in tqdm(sensor_paths):

    edf_path = os.path.join(device_path, sensor_path)

    csv_path = os.path.join(edf_path, 'headers.csv')

    file_list = [f for f in os.listdir(edf_path)
                 if f.lower().endswith('.edf') and not f.startswith('.')]
    file_list.sort()

    for file_name in tqdm(file_list):

        edf_reader = pyedflib.EdfReader(os.path.join(edf_path, file_name))
        edf_header = edf_reader.getHeader()
        edf_datarecord_duration = edf_reader.datarecord_duration
        edf_signal_headers = edf_reader.getSignalHeaders()
        edf_signals = [edf_reader.readSignal(sig_num) for sig_num in range(0, edf_reader.signals_in_file)]
        edf_reader.close()

        # set updates
        admin_code = 'OND06'
        edf_header.update({'admincode': admin_code})

        if edf_header['patientcode'] == '1031':
            edf_header.update({'patientcode': 2364})

        if edf_header['recording_additional'] == 'head':
            edf_header.update({'recording_additional': 'aid'})

        edf_writer = pyedflib.EdfWriter(os.path.join(edf_path, file_name), len(edf_signals))

        if any(edf_signal_header['sample_rate'] < 1 for edf_signal_header in edf_signal_headers):

            # loop through signals
            for sig in range(len(edf_signal_headers)):

                # adjust sample rate to samples per datarecord
                # (problem with pyedflib interpretation of 'sample_rate')
                old_sample_rate = edf_signal_headers[sig]['sample_rate']
                edf_signal_headers[sig].update({'sample_rate': edf_datarecord_duration * old_sample_rate})

                # set datarecord duration
                edf_writer.setDatarecordDuration(edf_datarecord_duration * 100000)

        edf_writer.setHeader(edf_header)
        edf_writer.setSignalHeaders(edf_signal_headers)
        edf_writer.writeSamples(edf_signals)
        edf_writer.close()

    edf.read_header(edf_path, csv_path)