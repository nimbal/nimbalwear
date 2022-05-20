from src.nimbalwear import Data

device = Data()
device.import_edf('W:/dev/autocalibration/uncalibrated/OND09_0128_01_AXV6_RWrist.edf')
device.autocal(plot=True)