import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
from pathlib import Path
import nwfiles.file.GENEActiv as ga
import nwfiles.file.EDF as edf

bin_dir = r'/Volumes/KIT_DATA/ReMiNDD/raw_data/GENEActiv/'
edf_dir = r'/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/standard_device_edf'
overwrite = False

Path(edf_dir).mkdir(parents=True, exist_ok=True)

csv_path = os.path.join(edf_dir, 'headers.csv')

ga.convert_dir_edf(bin_dir, edf_dir, correct_drift=True, deid=True, overwrite=overwrite, quiet=True)

edf.read_header(edf_dir, csv_path, quiet=True)
