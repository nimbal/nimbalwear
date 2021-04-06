import sys
sys.path.append(r'/Users/kbeyer/repos')

import nwpipeline.nwpipeline as gnac

file_patterns = ['*1027*LA*']

study_dir = '/Volumes/KIT_DATA/ReMiNDD'

convert_edf = True
separate_sensors = False
crop_nonwear = False

nonwear_csv = ('/Volumes/KIT_DATA/PD_DANCE_TWH/processed_data/GNAC/standard_nonwear_times/' +
               'GNAC_standard_nonwear_times.csv')

gnac.process_gnac(study_dir, file_patterns=file_patterns, nonwear_csv=nonwear_csv, convert_edf=convert_edf,
                  separate_sensors=separate_sensors, crop_nonwear=crop_nonwear, quiet=True)
