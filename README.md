# nwpipeline


nwpipeline is the the NiMBaLWear data processing pipeline. This pipeline is used to process data from wearable sensors.

This package is pre-release and should not be distributed outside the NiMBaLWear team. Additional functionality and documentation will be provided in subsequent releases.

# Contents

- `nwpipeline` is the actual nwpipeline package. This package contains `class NWPipeline`, which represents an instance of the pipeline and contains methods that process data and move it through the pipeline.

# Installation

To install nwpipeline using pip, run the following line in terminal or console:

`pip install git+https://github.com/nimbal/nwpipeline#egg=nwpipeline`

# Package Dependency

To include nwpipeline as a dependency in your Python package, include the following line in `setup.py` or include the string within the list alongside your other dependencies:

`install_requires=['nwpipeline@git+https://github.com/nimbal/nwpipeline#egg=nwpipeline']`

