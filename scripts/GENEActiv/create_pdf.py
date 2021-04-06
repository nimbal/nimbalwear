import sys
sys.path.append('/Users/kbeyer/repos')

from pathlib import Path
import os
from tqdm import tqdm
import nwfiles.file.GENEActiv as ga
from fnmatch import fnmatch

bin_path = '/Volumes/KIT_DATA/ReMiNDD/raw_data/GENEActiv'
pdf_path = '/Volumes/KIT_DATA/ReMiNDD/processed_data/GENEActiv/bin_pdf'

file_patterns = ['*5919*RW*','*5919*LW*']

Path(pdf_path).mkdir(parents=True, exist_ok=True)

file_list = [f for f in os.listdir(bin_path)
             if f.lower().endswith('.bin') and not f.startswith('.')
             and any([fnmatch(f, file_pattern) for file_pattern in file_patterns])]
file_list.sort()

for file_name in tqdm(file_list):

    ga_file = ga.GENEActivFile(os.path.join(bin_path, file_name))

    ga_file.read(parse_data=False, quiet=True)

    ga_file.create_pdf(pdf_path, correct_drift=False, quiet=True)
