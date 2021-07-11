import time
import datetime as dt

import nwdata
from nwsleep.sleep import *
import seaborn as sns
import matplotlib.pyplot as plt

# test-HANDDS

device_edf = '/Volumes/KIT_DATA/test-HANDDS/processed/cropped_device_edf/AXV6/test-HANDDS_060821_01_AXV6_LAnkle.edf'

nonwear_csv = '/Volumes/KIT_DATA/test-HANDDS/analyzed/nonwear/standard_nonwear_times/AXV6/test-HANDDS_060821_01_AXV6_LAnkle_NONWEAR.csv'
gait_csv = '/Volumes/KIT_DATA/test-HANDDS/analyzed/gait/gait_bouts/test-HANDDS_060821_01_GAIT_BOUTS.csv'
sptw_csv = '/Volumes/KIT_DATA/test-HANDDS/analyzed/sleep/sptw/test-HANDDS_060821_01_SPTW.csv'
sleep_csv = '/Volumes/KIT_DATA/test-HANDDS/analyzed/sleep/sleep_bouts/test-HANDDS_060821_01_SLEEP_BOUTS.csv'

# test_ReMiNDD

# device_edf = '/Volumes/KIT_DATA/test_ReMiNDD/processed/cropped_device_edf/GNAC/test_ReMiNDD_1027_01_GNAC_LA.edf'
#
# nonwear_csv = '/Volumes/KIT_DATA/test_ReMiNDD/analyzed/nonwear/standard_nonwear_times/GNAC/test_ReMiNDD_1027_01_GNAC_LA_NONWEAR.csv'
# gait_csv = '/Volumes/KIT_DATA/test_ReMiNDD/analyzed/gait/gait_bouts/test_ReMiNDD_1027_01_GAIT_BOUTS.csv'
# sptw_csv = '/Volumes/KIT_DATA/test_ReMiNDD/analyzed/sleep/sptw/test_ReMiNDD_1027_01_SPTW.csv'
# sleep_csv = '/Volumes/KIT_DATA/test_ReMiNDD/analyzed/sleep/sleep_bouts/test_ReMiNDD_1027_01_SLEEP_BOUTS.csv'

epoch_length = 5

print("Reading event files")

# import nonwear
nonwear = pd.read_csv(nonwear_csv, dtype=str)

nonwear['start_time'] = pd.to_datetime(nonwear['start_time'], format='%Y-%m-%d %H:%M:%S')
nonwear['end_time'] = pd.to_datetime(nonwear['end_time'], format='%Y-%m-%d %H:%M:%S')

# import sptw
sptw = pd.read_csv(sptw_csv, dtype=str)

sptw['start_time'] = pd.to_datetime(sptw['start_time'], format='%Y-%m-%d %H:%M:%S')
sptw['end_time'] = pd.to_datetime(sptw['end_time'], format='%Y-%m-%d %H:%M:%S')

# import sleep
sleep_bouts = pd.read_csv(sleep_csv, dtype=str)

sleep_bouts['start_time'] = pd.to_datetime(sleep_bouts['start_time'], format='%Y-%m-%d %H:%M:%S')
sleep_bouts['end_time'] = pd.to_datetime(sleep_bouts['end_time'], format='%Y-%m-%d %H:%M:%S')

# import gait
gait_bouts = pd.read_csv(gait_csv, dtype=str)

gait_bouts['start_time'] = pd.to_datetime(gait_bouts['start_timestamp'], format='%Y-%m-%d %H:%M:%S')
gait_bouts['end_time'] = pd.to_datetime(gait_bouts['end_timestamp'], format='%Y-%m-%d %H:%M:%S')

print("Reading device file")

# read device data
nwdevice = nwdata.NWData()
nwdevice.import_edf(device_edf)

start = time.time()

print("Wrangling")

start_datetime = nwdevice.header['startdate']

start_time = start_datetime+ dt.timedelta(hours=23)
end_time = start_datetime + dt.timedelta(hours=35)

nwdevice.crop(new_start_time=start_time, new_end_time=end_time)

x_accel = nwdevice.signals[nwdevice.get_signal_index('Accelerometer x')]
y_accel = nwdevice.signals[nwdevice.get_signal_index('Accelerometer y')]
z_accel = nwdevice.signals[nwdevice.get_signal_index('Accelerometer z')]

sample_rate = round(nwdevice.signal_headers[0]['sample_rate'])
start_datetime = nwdevice.header['startdate']


duration = time.time() - start

print(f"Duration: {duration} seconds")

# Plot z angle and events

timepoint = [start_datetime + dt.timedelta(seconds=(x / sample_rate)) for x in range(len(x_accel))]

start = time.time()

df = pd.DataFrame(data={"x": x_accel, "y": y_accel, "z": z_accel}, index=timepoint)

print("Plotting device")

sns.set_theme(style='white')
graph = sns.lineplot(data=df, linewidth=0.2)

print("Plotting events")

for index, row in nonwear.iterrows():
    graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=0.01, alpha=0.5, color='red')

for index, row in sptw.iterrows():
    graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0.015, ymax=0.025, alpha=0.5, color='grey')

for index, row in sleep_bouts.iterrows():
    graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0.03, ymax=0.04, alpha=0.5, color='cyan')

for index, row in gait_bouts.iterrows():
    graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0.045, ymax=0.055, alpha=0.5, color='green')

duration = time.time() - start

print(f"Duration: {duration} seconds")
