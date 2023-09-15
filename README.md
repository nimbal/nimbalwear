# nimbalwear

nimbalwear is an open source toolkit for processing data from wearable sensors.

This package is pre-release and should not be distributed outside the NiMBaLWear team. Additional functionality and 
documentation will be provided in subsequent releases.

# Contents

- Under construction

# Installation

To install the latest release of nimbalwear directly from GitHub using pip, run the following line in terminal or 
console:

`pip install git+https://github.com/nimbal/nimbalwear`

To install a specific release, insert `M.m` after the repository name to install from the branch associated with that 
minor release. For example:

`pip install git+https://github.com/nimbal/nimbalwear@0.18`

# Package Dependency

To include the latest release of nimbalwear as a dependency in your Python package, include the following line in 
`setup.py` or include the string within the list alongside your other dependencies:

`install_requires=['nimbalwear@git+https://github.com/nimbal/nimbalwear@[version]']`

To include a specific release, replace `[version]` with the branch associated with that minor release.

# Changes by version

v0.21.3
- update vertdetach version
- update compatibility with pyedflib v1.0.34 (sex header field and sample_frequency)
- bug fix: datetime conversion when reading nonwear csv

v0.21.2
- bug fix: indexing issue caused states to sometimes be skipped
- bug fix: start date calculation for multiple gait devices

v0.21.1
- fixed MANIFEST.in bug

v0.21.0
- reorganized gait module code

v0.20.1
- adjust start time moved to before sync

v0.20.0
- add autocal offset and scale outputs
- option to save separate sensor EDF files after data prep
- move settings dump from log into separate file
- rename Pipeline class to Study
- separate default, study, and custom pipeline settings
- some missing data handled and reported as warning instead of raising exception
- bug fix: all filters now dual pass
- separate sync event and segments into separate folders
- syncs detected from any axis rather than choosing those from one axis
- include config sync in sync list
- add ref device type and location to sync output


v0.19.8
- bug fix: properly handles Axivity devices with no gyro collected

v0.19.7
- bug fix: instead of error, sync returns empty DataFrame when no syncs detected or matched
- bug fix: fix bug where sync plots aren't displayed

v0.19.6
- bug fix: ensure physical_min < physical_max when writing edf

v0.19.5
- bug fix: properly detects sleep bouts when entire SPTW is sleep

v0.19.4
- added utility to read password protected excel files

v0.19.3
- bug fix: adjust filter order in activity module
- bug fix: fix error when trying to run activity module with no sleep detected

v0.19.2
- bug fix: properly calculates sample indices to be removed - no longer attempts to remove sample beyond end of window

v0.19.1
- add minimum correlation option for sync
- bug fix: references to settings.json instead of settings.toml on install

v0.19.0
- subjects.csv renamed to collections.csv and coll_id column added
- settings.toml replaces settings.json
- moved config_time check to only occur on first device in sync and display appropriately in log
- activity analysis now done for all wrist devices available with options to select cutpoints
- Pipeline.add_custom_events() provides ability to add or replace custom events from csv 
- can specify separate non-wear detection parameters for ankle, wrist, and trunk devices

v0.18.3
- change search radius to minutes
- fix bug in plot if sync is near end of sync radius
- fix assignment bug if search radius is not set

v0.18.2
- fix bug where pipeline tries to autocalibrate data from file that wasn't found

v0.18.1
- fix bug when sync search radius falls outside target collection time

v0.18.0
- insert wear bouts between nonwear bouts (added event column)

v0.17.3
- fixed bug excluding sleep windows to exclude from activity

v0.17.2
- accel step detection uses gait_stats for summary

v0.17.1
- rename start_timestamp and end_timestamp to start_time and end_time in gait

v0.17.0
- rename "feedback report"
- add daily non-wear summary
- crop non-wear from start of collection
- separate sedentary detected from wrist data while walking from other sedentary
- option to classify sptw and sleep bouts as 'overnight'

v0.16.3
- handle Bittium file import if header is imperfect
- add sync search radius

v0.16.2
- activity fixes
  - output 1-second avm
  - fix hard-coded epoch_length

v0.16.1
- add accel_std_thresh_mg as nonwear setting in JSON file
- check for accelerometer signals before autocal
- fix Nonin device data import bugs
- fix missing gait pushoff data bug

v0.16.0
- updated filtering and vm calculation for activity calculation (faster)
- renamed Data object to Device
- added autocalibration of accelerometers
- added relevant functions from  nwdata, nwnonwear, nwgait, nwsleep, nwactivity, and nwapp as data.py, nonwear.py, 
gait.py, sleep.py, activity.py and app.py modules
- tidy sync outputs
- add option to provide alternative settings.json file
- new processing log for each collection
- output settings to log

v0.15.2
- update to nwdata v0.9.2 (much faster file reading and writing)
- update to nwgait v0.4.2
- new processing log for each run call

v0.15.1
- fix package setup files

v0.15.0
- add option to adjust start time of device on convert
- update to nwnonwear v0.2.0 that uses vertdetach package
- allow choice between accel and gyro step detection
- update pandas append to concat

v0.14.2
- bug fix: non-wear end detection windows in proper direction (nwnonwear v0.1.3)
- bug fix: non-wear detection accounts for temperature frequency in rate of change (nwnonwearv0.1.3)

v0.14.1
- add option to lowpass data before activity calculations (nwactivity v0.2.1)
- bug fix: does not count activity during nonwear or sleep (nwactivity v0.3.0)

v0.14.0
- add option to synchronize devices on convert (nwdata v0.9.0)
- select activity cutpoints based on age (nwactivty v0.2.0)
- bug fix: dominant hand from subjects.csv no longer case sensitive

v0.13.0
- gyro step detection for gait

v0.12.0
- nwdata v0.8.0 update
  - add option to crop NWData inplace
  - add read_header method to CWAFile
  - add rotate_z method to rotate accelerometer and gyroscope data around z axis
  - adjust Bittium accelerometer signals to g instead of mg
  - restructure NWData header
  
v0.11.1
- create separate file for rejected steps
- bug fix: only include single device in cropped nonwear csv
- bug fix: report correct steps detected in log file
- bug fix: do not allow sleep and nonwear to overlap (nwsleep v0.3.1)
- bug fix: dedicated logger based on study code

v0.11.0
- reorganized Pipeline and Collection classes
- moved many settings to settings.json file
- read and convert split so convert can be tracked by status
- update to nwdata v0.7.2
- adjust gait algorithm and vertical axis detection (nwgait v0.3.0)  
- bug fix: quiet variable is passed to edf export function

v0.10.0
- modify collection loop to only perform collections included in device list
- create cropped non-wear time file
- bug fix: file duration calculation while cropping (nwdata v0.7.1)

v0.9.0
- updated to nwdata v0.7.0 to incorporate multiple changes
  - convert to ndarray  before write with pyedflib (bug)
  - handle Bittium Faros 360 and variable signals
  - update device type codes
- Modify device and sensor logic to match nwdata v0.7.0
- bug fix: axis selection during gait detection (nwgait v0.1.4)

v0.8.0
- add option to run daily sleep stats on all sptw that contain sleep (nwsleep v0.3.0)

v0.7.2
- add option to select axis used to detect gait
- add daily_all sleep stats output
- bug fix: convert device locations to upper case when selecting devices
- bug fix: add column names to steps table if none found (workaround)
- bug fix: check for minimum usable data and candidate sptw and sleep bouts before continuing processing (nwsleep v0.2.2)

v0.7.1
- update pyedflib for all required packages

v0.7.0
- add mechanism to add data dictionaries to output folders
- ignore non-wear when detecting sleep period time windows (nwsleep v0.2.0)
- run t5a5 and t8a4 sleep bout detection
- require pyedflib v0.1.22

v0.6.0
- add non-wear detection for Axivity devices (AXV6) (nwnonwear v0.1.2)
- interpolate inserted values when correcting clock drift or sample rate (nwdata v0.5.0)
- faster GENEActiv read and progress bars on clock drift correct (nwdata v0.6.0)
- adjust device selection logic for activity, gait, sleep analytics

v0.5.0
- update `nwdata` to require v0.4.0
    - fixes bug where startdate not updated when cropping NWData
    - adds method to get day indices of a signal
- add subjects.csv
- add sleep detection and analysis (nwsleep v0.1.0)
- fix bug where errors during collection not logged correctly

v0.4.0
- update `nwactivity` to require v0.1.1 (remove mvpa from daily summary)
- update `nwgait` to require v0.1.2 (add daily gait summary)
- add daily gait summary output
- add pipeline status tracking
- renames EDF files to standard names based on information from devices.csv
- loads data only required devices for single stage
- add support for Axivity AX6 devices (update `nwdata` to require v0.3.0)

v0.3.0
- add gait processing (nwgait v0.1.0)
- add unexpected error handling with traceback output to log
- add activity processing (nwactivity v0.1.0)
- add study_code as an identifier for all generated data
- add nonwear_bout_id to nonwear output (nwnonwear v0.1.1)

v0.2.0
- add nonwear processing (nwnonwear v0.1.0)
- add option to process only a single stage of the pipeline

v0.1.1
- update `nwdata` to require v0.1.2