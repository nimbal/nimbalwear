# nwpipeline


nwpipeline is the the NiMBaLWear data processing pipeline. This pipeline is used to process data from wearable sensors.

This package is pre-release and should not be distributed outside the NiMBaLWear team. Additional functionality and documentation will be provided in subsequent releases.

# Contents

- `nwpipeline` is the actual nwdata package. This package contains `class NWData`, which is used to represent and manipulate data within a structure compatible with the European Data Format (EDF). This class contains methods for importing data from a variety of devices based on the modules in the `nwfiles` subpackage. It also contains a method for exporting data to an EDF file.

# Installation

To install nwdata using pip, run the following line in terminal or console:

`pip install git+https://github.com/nimbal/nwpipeline#egg=nwpipeline`

# Package Dependency

To include nwdata as a dependency in your Python package, include the following line in `setup.py` or include the string within the list alongside your other dependencies:

`install_requires=['nwpipeline@git+https://github.com/nimbal/nwpipeline#egg=nwpipeline']`

