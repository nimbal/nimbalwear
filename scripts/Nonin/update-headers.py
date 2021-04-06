import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
import pandas as pd
import pyedflib
import nwfiles.file.EDF as edf

pkg_path = r'/Volumes/KIT_DATA/ReMiNDD/processed_data/Nonin/standard_edf'
snsr_info_path = r'/Volumes/KIT_DATA/ReMiNDD/processed_data/Tables/OND06_SNSR_INFO.csv'
csv_path = os.path.join(pkg_path, 'headers.csv')

snsr_info = pd.read_csv(snsr_info_path, dtype=str, na_filter=False)

file_list = [f for f in os.listdir(pkg_path) if f.lower().endswith('.edf')]
file_list.sort()

for file_name in file_list:

    print('Updating header for %s ...' % file_name)

    edf_reader = pyedflib.EdfReader(os.path.join(pkg_path, file_name))
    edf_header = edf_reader.getHeader()
    edf_signal_headers = edf_reader.getSignalHeaders()
    edf_signals = [edf_reader.readSignal(sig_num) for sig_num in range(0, edf_reader.signals_in_file)]
    edf_reader.close()

    admin_code = 'OND06'
    subject_id = 'OND06_SBH_' + edf_header['patientcode']

    device_id = snsr_info['snsr_base_slp_wrist'].loc[
        snsr_info['subject_id'] == subject_id].item()

    edf_header.update({'admincode': admin_code,
                       'equipment': 'Nonin_' + device_id,
                       'technician': ''})

    edf_writer = pyedflib.EdfWriter(os.path.join(pkg_path, file_name), len(edf_signals))
    edf_writer.setHeader(edf_header)
    edf_writer.setSignalHeaders(edf_signal_headers)
    edf_writer.writeSamples(edf_signals)
    edf_writer.close()

edf.read_edf_header(pkg_path, csv_path)