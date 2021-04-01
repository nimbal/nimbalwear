import sys
sys.path.append(r'/Users/kbeyer/repos')

import os
import nwfiles.file.EDF as edf

edf_path = ('/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/cropped_sensor_edf/' +
            'TEMPERATURE')

csv_path = os.path.join(edf_path, 'edf-headers.csv')

edf.header_summary(edf_path, csv_path)