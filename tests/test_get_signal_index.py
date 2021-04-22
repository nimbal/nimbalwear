import nwdata

file_path = '/Volumes/KIT_DATA/test_study/raw/BITF/OND06_SBH_8600_01_SE01_GABL_BF.EDF'

test_file = nwdata.NWData()
test_file.import_bitf(file_path)

label = 'Accelerometer x'

index = test_file.get_signal_index(label)

print(index)

label = 'Accelerometer y'

index = test_file.get_signal_index(label)

print(index)

label = 'Accelerometer z'

index = test_file.get_signal_index(label)

print(index)

label = 'ECG'

index = test_file.get_signal_index(label)

print(index)

label = 'Temperature'

index = test_file.get_signal_index(label)

print(index)
