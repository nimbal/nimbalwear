# nwpipeline

nwpipeline is the the NiMBaLWear data processing pipeline. This pipeline is used to process data from wearable sensors.

This package is pre-release and should not be distributed outside the NiMBaLWear team. Additional functionality and documentation will be provided in subsequent releases.

# Contents

- `nwpipeline` is the actual nwpipeline package. This package contains `class NWPipeline`, which represents an instance of the pipeline and contains methods that process data and move it through the pipeline.

# Installation

To install the latest release of nwpipeline directly from GitHub using pip, run the following line in terminal or console:

`pip install git+https://github.com/nimbal/nwpipeline@latest#egg=nwpipeline`

To install a specific release, replace `@latest` with the tag associated with that release. 

# Package Dependency

To include the latest release of nwpipeline as a dependency in your Python package, include the following line in `setup.py` or include the string within the list alongside your other dependencies:

`install_requires=['nwpipeline@git+https://github.com/nimbal/nwpipeline@latest#egg=nwpipeline']`

To include a specific release, replace `@latest` with the tag associated with that release.

# Changes by version

v0.2.0
- add nonwear processing (nwnonwear v0.1.0)

v0.1.1
- update `nwdata` to require v0.1.2