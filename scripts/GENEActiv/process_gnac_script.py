
import nwpipeline.NWPipeline as gnac

file_patterns = ['*1027*LA*']

study_dir = '/Volumes/KIT_DATA/test_study'

convert_edf = False
separate_sensors = False
crop_nonwear = True

nonwear_csv = ('/Volumes/KIT_DATA/test_study/analyzed_data/nonwear/standard_nonwear_times/' +
               'GNAC_standard_nonwear_times.csv')

gnac.process_gnac(study_dir, file_patterns=file_patterns, nonwear_csv=nonwear_csv, convert_edf=convert_edf,
                  separate_sensors=separate_sensors, crop_nonwear=crop_nonwear, quiet=True)
