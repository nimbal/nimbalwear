import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
from tqdm import tqdm
from pathlib import Path
import nwfiles.file.EDF as edf

in_dir = '/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_device_edf'

out_dirs = ['/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_sensor_edf/ACCELEROMETER',
            '/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_sensor_edf/TEMPERATURE',
            '/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_sensor_edf/LIGHT',
            '/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_sensor_edf/BUTTON']

out_channels = [[0, 1, 2],
                [3],
                [4],
                [5]]

for out_dir in out_dirs:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

file_list = [f for f in os.listdir(in_dir)
             if f.lower().endswith('.edf') and not f.startswith('.')]
file_list.sort()

for file_name in tqdm(file_list):

    in_path = os.path.join(in_dir, file_name)
    out_paths = [os.path.join(out_dir, file_name) for out_dir in out_dirs]

    edf.separate_sensors(in_path, out_paths, out_channels)
