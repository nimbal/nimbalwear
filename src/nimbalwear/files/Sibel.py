import os
import csv
import time


class SibelFile:

    def __init__(self, file_path):

        self.file_path = os.path.abspath(file_path)
        self.file_name = os.path.basename(self.file_path)
        self.file_dir = os.path.dirname(self.file_path)

        self.data = {}

    def read(self, quiet=False):

        read_start_time = time.time()

        # if file does not exist then exit
        if not os.path.exists(self.file_path):
            print(f"****** WARNING: {self.file_path} does not exist.\n")
            return

        # Read Nonin .asc file
        if not quiet:
            print("Reading %s ..." % self.file_path)

        samples = []

        # read file
        with open(self.file_path, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for line in reader:
                samples.append(dict(line))

        # combine dicts into one
        for key in samples[0].keys():
            self.data[key] = [d[key] for d in samples]

        if not quiet:
            print("Done reading file. Time to read file: ", time.time() - read_start_time, "seconds.")

