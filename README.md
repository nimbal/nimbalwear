# nwpipeline

nwpipeline is the the NiMBaLWear data processing pipeline. This pipeline is used to process data from wearable sensors.

This package is pre-release and should not be distributed outside the NiMBaLWear team. Additional functionality and documentation will be provided in subsequent releases.

# Contents

- `nwpipeline` is the actual nwpipeline package. This package contains `class NWPipeline`, which represents an instance of the pipeline and contains methods that process data and move it through the pipeline.

# Installation

To install the latest release of nwpipeline directly from GitHub using pip, run the following line in terminal or console:

`pip install git+https://github.com/nimbal/nwpipeline#egg=nwpipeline`

To install a specific release, insert `@v#.#.#` after the repository name replacing with the tag associated with that release. For example:

`pip install git+https://github.com/nimbal/nwpipeline@v1.0.0#egg=nwpipeline`

# Package Dependency

To include the latest release of nwpipeline as a dependency in your Python package, include the following line in `setup.py` or include the string within the list alongside your other dependencies:

`install_requires=['nwpipeline@git+https://github.com/nimbal/nwpipeline@[version]#egg=nwpipeline']`

To include a specific release, replace `[version]` with the tag associated with that release.

# Changes by version

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