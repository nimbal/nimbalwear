import sys
sys.path.append(r'/Users/kbeyer/repos')

import nwfiles.file.Nonin as nonin
import nwfiles.file.EDF as edf
import os

asc_dir = r'/Volumes/KIT_DATA/ReMiNDD/raw_data/Nonin'

edf_dir = r'/Volumes/KIT_DATA/ReMiNDD/processed_data/Nonin/standard_edf'

csv_path = os.path.join(edf_dir, 'headers.csv')

nonin.convert_dir_edf(asc_dir, edf_dir, deid=True)

edf.read_edf_header(edf_dir, csv_path)

