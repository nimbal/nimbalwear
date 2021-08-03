# nwpipeline

nwpipeline is the the NiMBaLWear data processing pipeline. This pipeline is used to process data from wearable sensors.

This package is pre-release and should not be distributed outside the NiMBaLWear team. Additional functionality and documentation will be provided in subsequent releases.

# Contents

- `nwpipeline` is the actual nwpipeline package. This package contains `class NWPipeline`, which represents an instance of the pipeline and contains methods that process data and move it through the pipeline.

# Installation

To install the latest release of nwpipeline directly from GitHub using pip, run the following line in terminal or console:

`pip install git+https://github.com/nimbal/nwpipeline#egg=nwpipeline`

To install a specific release, replace `@v#.#.#` with the tag associated with that release. For example:

`pip install git+https://github.com/nimbal/nwpipeline@v1.0.0#egg=nwpipeline`

# Package Dependency

To include the latest release of nwpipeline as a dependency in your Python package, include the following line in `setup.py` or include the string within the list alongside your other dependencies:

`install_requires=['nwpipeline@git+https://github.com/nimbal/nwpipeline@latest#egg=nwpipeline']`

To include a specific release, replace `@latest` with the tag associated with that release.

# Changes by version

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