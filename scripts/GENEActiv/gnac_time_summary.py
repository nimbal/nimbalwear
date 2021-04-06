import sys
sys.path.append('/Users/kbeyer/repos')

import pandas as pd
import datetime as dt
import os
from tqdm import tqdm
import nwfiles.file.GENEActiv as ga

folder_path = r'/Volumes/KIT_DATA/PD_DANCE_TWH/raw_data/GNAC'
csv_name = 'time_summary.csv'

summary_df = pd.DataFrame(columns=['subject', 'device_location', 'samples', 'sample_rate', 'duration', 'clock_drift',
                                   'clock_drift_rate', 'config_time', 'start_time', 'page_start_time', 'end_time',
                                   'adj_end_time', 'page_end_time', 'adj_page_end_time', 'extract_time', 'file'])

file_list = [f for f in os.listdir(folder_path)
             if f.lower().endswith('.bin') and not f.startswith('.')]
file_list.sort()

print('Reading files ...')

for file_name in tqdm(file_list):

    ga_file = ga.GENEActivFile(os.path.join(folder_path, file_name))

    ga_file.read(end=1, quiet=True)

    samples = ga_file.samples
    sample_rate = ga_file.data['sample_rate']
    duration = dt.timedelta(seconds=samples/sample_rate)
    clock_drift = float(ga_file.header['Extract Notes'].split(' ')[3][:-2].replace(',', ''))
    clock_drift_rate = ga_file.clock_drift_rate
    config_time = dt.datetime.strptime(ga_file.header['Config Time'], '%Y-%m-%d %H:%M:%S:%f')
    start_time = dt.datetime.strptime(ga_file.header['Start Time'], '%Y-%m-%d %H:%M:%S:%f')
    page_start_time = ga_file.data['start_time']
    end_time = start_time + duration
    page_end_time = page_start_time + duration
    extract_time = dt.datetime.strptime(ga_file.header['Extract Time'], '%Y-%m-%d %H:%M:%S:%f')
    adj_end_time = end_time - ((end_time - config_time) * clock_drift_rate)
    adj_page_end_time = page_end_time - ((page_end_time - config_time) * clock_drift_rate)

    summary_df = summary_df.append({'subject': ga_file.header['Subject Code'],
                                    'device_location': ga_file.header['Device Location Code'],
                                    'samples': samples,
                                    'sample_rate': sample_rate,
                                    'duration': duration,
                                    'clock_drift': clock_drift,
                                    'clock_drift_rate': clock_drift_rate,
                                    'config_time': config_time,
                                    'start_time': start_time,
                                    'page_start_time': page_start_time,
                                    'end_time': end_time,
                                    'adj_end_time': adj_end_time,
                                    'page_end_time': page_end_time,
                                    'adj_page_end_time': adj_page_end_time,
                                    'extract_time': extract_time,
                                    'file': file_name},
                                   ignore_index=True)

summary_df.to_csv(os.path.join(folder_path, csv_name), index=False)
