import sys
sys.path.append(r'C:\Users\kitbeyer\repos')

import os
import shutil

path = r'C:\Users\kitbeyer\data\Nonin\raw_data'

file_list = [f for f in os.listdir(path) if f.lower().endswith('.asc')]
file_list.sort()

for file_name in file_list:
    file_base, file_ext = os.path.splitext(file_name)
    file_base_parts = file_base.split('_')
    new_file_name = '_'.join(file_base_parts[:6] + ['NONW'])

    file_num = file_base_parts[7] if len(file_base_parts) == 8 else '01'
    new_file_name = new_file_name + '_' + file_num + file_ext

    old_file_path = os.path.join(path, file_name)
    new_file_path = os.path.join(path, new_file_name)

    shutil.copy(old_file_path, new_file_path)
