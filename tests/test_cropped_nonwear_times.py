import pandas as pd


nonwear_csv_path = '/Volumes/KIT_DATA/test-HANDDS/analyzed/nonwear/standard_nonwear_times/AXV6/test-HANDDS_060821_01_AXV6_LAnkle_NONWEAR.csv'

study_code = 'test-HANDDS'
subject_id = '060821'
coll_id = '01'
device_type = 'AXV6'
device_location = 'LAnkle'

nonwear_times = pd.DataFrame()

# read nonwear csv file
nonwear_times = pd.read_csv(nonwear_csv_path, dtype=str)
nonwear_times['start_time'] = pd.to_datetime(nonwear_times['start_time'], format='%Y-%m-%d %H:%M:%S')
nonwear_times['end_time'] = pd.to_datetime(nonwear_times['end_time'], format='%Y-%m-%d %H:%M:%S')

# append to collection attribute
nonwear_times = nonwear_times.append(nonwear_times, ignore_index=True)

# get last device nonwear period

device_nonwear_idx = nonwear_times.index[(nonwear_times['study_code'] == study_code) &
                                         (nonwear_times['subject_id'] == subject_id) &
                                         (nonwear_times['coll_id'] == coll_id) &
                                         (nonwear_times['device_type'] == device_type) &
                                         (nonwear_times['device_location'] == device_location)].tolist()

last_nonwear_idx = device_nonwear_idx[-1]

last_nonwear = nonwear_times.loc[last_nonwear_idx]


print(nonwear_times)
print(last_nonwear_idx)
print(last_nonwear)

nonwear_times.drop(index=last_nonwear_idx, inplace=True)

print(nonwear_times)