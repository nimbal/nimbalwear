from pathlib import Path
import json

import pandas as pd
import numpy as np
from nwdata import NWData
from tqdm import tqdm
import nwactivity

study_save_dir = 'W:/NiMBaLWEAR/'
study_code = 'OND09'

csv_out = 'W:/dev/activity_files/uncalibrated/uncalib_'

# subject_id = '0059'
# coll_id = '01'
# device_type='AXV6'
# device_location='RWrist'

epoch_secs = 12


# Read metadata files
study_save_dir = Path(study_save_dir)
study_dir = study_save_dir / study_code
settings_path = study_dir / 'pipeline/settings.json'
devices_csv_path = study_dir / 'pipeline/devices.csv'


with open(settings_path, 'r') as f:
    settings_json = json.load(f)

dirs = settings_json['pipeline']['dirs']
dirs = {key: study_dir / value for key, value in dirs.items()}

devices_csv = pd.read_csv(devices_csv_path, dtype=str)

pbar = tqdm(devices_csv.iterrows(), total=devices_csv.shape[0])

for idx, row in pbar:

    subject_id = row['subject_id']
    coll_id = row['coll_id']
    device_type = row['device_type']
    device_id = row['device_id']
    device_location = row['device_location']

    if (device_location == "RWrist") | (device_location == "LWrist"):
        # Concatenate data file paths
        device_edf_path = (dirs['device_edf_standard']
                       / ("_".join([study_code, subject_id, coll_id, device_type, device_location]) + ".edf"))

        # read accelerometer data file
        device = NWData()
        device.import_edf(device_edf_path, quiet=True)

        # get accelerometer x, y, z signals
        accel_x_sig = device.get_signal_index('Accelerometer x')
        accel_y_sig = device.get_signal_index('Accelerometer y')
        accel_z_sig = device.get_signal_index('Accelerometer z')

        x = device.signals[accel_x_sig]
        y = device.signals[accel_y_sig]
        z = device.signals[accel_z_sig]

        sample_rate = device.signal_headers[accel_x_sig]['sample_rate']
        start_datetime = device.header['start_datetime']

        avm = nwactivity.activity_wrist_avm(x, y, z, sample_rate, start_datetime)

        activity_epochs = avm[0]

        stats = nwactivity.activity_stats(activity_epochs, quiet=True)

        csv_out_path = Path(csv_out + ("_".join([study_code, subject_id, coll_id, device_type, device_location]) + ".csv"))

        stats.to_csv(csv_out_path, index=False)