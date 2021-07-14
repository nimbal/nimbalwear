import time
import datetime as dt
import os

import nwdata
from nwsleep.sleep import *
import seaborn as sns
import matplotlib.pyplot as plt

study_dir = os.path.abspath(r'W:\NiMBaLWEAR\test-HANDDS')

device_edf = os.path.join(study_dir, r'processed\standard_device_edf\AXV6\test-HANDDS_061621_01_AXV6_RWrist.edf')

nonwear_csv = os.path.join(study_dir, r'analyzed\nonwear\standard_nonwear_times\AXV6\test-HANDDS_061621_01_AXV6_RWrist_NONWEAR.csv')
gait_csv = os.path.join(study_dir, r'analyzed\gait\gait_bouts\test-HANDDS_061621_01_GAIT_BOUTS.csv')
#sptw_csv = os.path.join(study_dir, r'analyzed\sleep\sptw\test-HANDDS_061621_01_SPTW.csv')
#sleep_csv = os.path.join(study_dir, r'analyzed\sleep\sleep_bouts\test-HANDDS_061621_01_SLEEP_BOUTS.csv')

epoch_length = 5

print("Reading event files")

# import nonwear
nonwear = pd.read_csv(nonwear_csv, dtype=str)

nonwear['start_time'] = pd.to_datetime(nonwear['start_time'], format='%Y-%m-%d %H:%M:%S')
nonwear['end_time'] = pd.to_datetime(nonwear['end_time'], format='%Y-%m-%d %H:%M:%S')

# # import sptw
# sptw = pd.read_csv(sptw_csv, dtype=str)
#
# sptw['start_time'] = pd.to_datetime(sptw['start_time'], format='%Y-%m-%d %H:%M:%S')
# sptw['end_time'] = pd.to_datetime(sptw['end_time'], format='%Y-%m-%d %H:%M:%S')
#
# # import sleep
# sleep_bouts = pd.read_csv(sleep_csv, dtype=str)
#
# sleep_bouts['start_time'] = pd.to_datetime(sleep_bouts['start_time'], format='%Y-%m-%d %H:%M:%S')
# sleep_bouts['end_time'] = pd.to_datetime(sleep_bouts['end_time'], format='%Y-%m-%d %H:%M:%S')

# import gait
gait_bouts = pd.read_csv(gait_csv, dtype=str)

gait_bouts['start_time'] = pd.to_datetime(gait_bouts['start_timestamp'], format='%Y-%m-%d %H:%M:%S')
gait_bouts['end_time'] = pd.to_datetime(gait_bouts['end_timestamp'], format='%Y-%m-%d %H:%M:%S')

print("Reading device file")

# read device data
nwdevice = nwdata.NWData()
nwdevice.import_edf(device_edf)

print("Wrangling")

start = time.time()

x_accel = nwdevice.signals[nwdevice.get_signal_index('Accelerometer x')]
y_accel = nwdevice.signals[nwdevice.get_signal_index('Accelerometer y')]
z_accel = nwdevice.signals[nwdevice.get_signal_index('Accelerometer z')]

sample_rate = round(nwdevice.signal_headers[0]['sample_rate'])
start_datetime = nwdevice.header['startdate']

epoch_samples = sample_rate * epoch_length
z_sample_rate = 1 / epoch_length

print("Calculating z-angle")

z_angle, z_angle_diff = z_angle_change(x_values=x_accel, y_values=y_accel, z_values=z_accel, epoch_samples=epoch_samples)

duration = time.time() - start

print(f"Duration: {duration} seconds")

# Plot z angle and events

timepoint = [start_datetime + dt.timedelta(seconds=(x / z_sample_rate)) for x in range(len(z_angle_diff))]

start = time.time()

z_df = pd.DataFrame(data={'z_angle': z_angle[:-1], 'z_angle_diff': z_angle_diff}, index=timepoint)

print("Plotting device")

sns.set_theme(style='white')
graph = sns.lineplot(data=z_df, linewidth=0.2)

# graph.axhline(5, color='black')

print("Plotting events")

for index, row in nonwear.iterrows():
    graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0, ymax=0.01, alpha=0.5, color='red')

# for index, row in sptw.iterrows():
#     graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0.015, ymax=0.025, alpha=0.5, color='grey')

# for index, row in sleep_bouts.iterrows():
#     graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0.03, ymax=0.04, alpha=0.5, color='cyan')
#
for index, row in gait_bouts.iterrows():
    graph.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymin=0.045, ymax=0.055, alpha=0.5, color='green')

duration = time.time() - start

print(f"Duration: {duration} seconds")
