#
# CWA structure found at: https://github.com/digitalinteraction/openmovement/blob/master/Docs/ax3/cwa.h
# https://github.com/digitalinteraction/openmovement/blob/master/Docs/ax3/ax3-technical.md

# Code adapted from:
# https://github.com/digitalinteraction/openmovement/blob/master/Software/AX3/cwa-convert/python/cwa_metadata.py
# https://github.com/arsalikhov/read_cwa/blob/main/scripts/AxivityFile.py
# https://github.com/cran/GGIR/blob/master/R/g.cwaread.R
#
# Kit Beyer, 2021

import sys
import os
import time
from datetime import datetime
from datetime import timedelta
from struct import unpack

import numpy as np

from .EDF import EDFFile

# TODO: Only works for AX6 - add functionality for packing_format = 0 and mags
# TODO: add various checks and error handling


class CWAFile:

    def __init__(self, file_path):

        self.file_path = os.path.abspath(file_path)
        self.file_name = os.path.basename(self.file_path)
        self.file_dir = os.path.dirname(self.file_path)
        self.header = {}
        self.data = {}
        self.file_size = None

    def read_header(self, quiet=False):

        read_start_time = time.time()

        # if file does not exist then exit
        if not os.path.exists(self.file_path):
            print(f"****** WARNING: {self.file_path} does not exist.\n")
            return

        # Read GENEActiv .bin file
        if not quiet:
            print("Reading %s ..." % self.file_path)

        self.file_size = os.path.getsize(self.file_path)

        with open(self.file_path, 'rb') as cwa_file:

            if not quiet:
                print("Reading header packet ...")

            header_packet = cwa_file.read(1024)
            self.header = self.parse_header_packet(header_packet)

    def read(self, resample=True, quiet=False):

        read_start_time = time.time()

        # if file does not exist then exit
        if not os.path.exists(self.file_path):
            print(f"****** WARNING: {self.file_path} does not exist.\n")
            return

        # Read GENEActiv .bin file
        if not quiet:
            print("Reading %s ..." % self.file_path)

        self.file_size = os.path.getsize(self.file_path)

        if not quiet:
            print("Reading file ...")

        dtype = np.dtype('B')
        with open(self.file_path, 'rb') as f:
            data_bytes = np.fromfile(f, dtype)

        if not quiet:
            print("Parsing header ...")

        header_packet = data_bytes[:1024].tobytes()
        self.header = self.parse_header_packet(header_packet)

        if not quiet:
            print("Parsing data ...")

        data_packets = data_bytes[1024:].reshape(-1, 512)
        data_header_packets = data_packets[:, :30]
        data_samples = data_packets[:, 30:510].tobytes()

        data_headers = [self.parse_data_header(dh.tobytes()) for dh in data_header_packets]

        num_axes = data_headers[0][1]
        packing_format = data_headers[0][3]
        accel_scale = data_headers[0][4]
        gyro_scale = data_headers[0][5]

        axes = []
        scale = []

        if num_axes == 3:
            axes = ['ax', 'ay', 'az']
            scale = [accel_scale, accel_scale, accel_scale]
        elif num_axes == 6:
            axes = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
            scale = [gyro_scale, gyro_scale, gyro_scale, accel_scale, accel_scale, accel_scale]

        if packing_format == 2:
            data = np.array(unpack(f'{int(len(data_samples) / 2)}h', data_samples)).reshape(-1, num_axes).T
            self.data = {ax: (data[i] * scale[i]) for i, ax in enumerate(axes)}
            self.data['light'] = np.array([dh[6] for dh in data_headers])
            self.data['temperature'] = np.array([dh[7] for dh in data_headers])
            self.data['battery'] = np.array([dh[8] for dh in data_headers])

        samples_per_packet = data_headers[0][2]
        first_start_time = data_headers[0][0]
        last_start_time = data_headers[-1][0]
        num_packets = len(data_headers)
        actual_elapsed = last_start_time - first_start_time
        actual_sample_rate = ((num_packets - 1) * samples_per_packet) / actual_elapsed.total_seconds()

        self.header['start_time'] = first_start_time

        if resample:

            if not quiet:
                print("Resampling data ...")

            samples_per_second_diff = actual_sample_rate - self.header['sample_rate']
            # drift_rate = samples_per_second_diff / actual_sample_rate
            adjust_rate = abs(actual_sample_rate / samples_per_second_diff)

            if samples_per_second_diff > 0:  # if drift is positive then remove extra samples

                for key in self.data.keys():
                    # delete data from each signal
                    self.data[key] = np.delete(self.data[key],
                                               [int(adjust_rate * (i + 1))
                                                for i in range(int(len(self.data[key]) / adjust_rate))])

                    self.data[key] = self.data[key]

            else:  # else add samples

                for key in self.data.keys():

                    insert_count = int(len(self.data[key]) / adjust_rate)
                    insert_before = [int(adjust_rate * i) for i in range(1, insert_count)]
                    insert_value = [(self.data[key][i - 1] + self.data[key][i]) / 2 for i in insert_before]

                    # insert data into each signal
                    self.data[key] = np.insert(self.data[key], insert_before, insert_value)

                    self.data[key] = self.data[key]

        else:

            self.header['sample_rate'] = actual_sample_rate

        self.header['packet_rate'] = self.header['sample_rate'] / samples_per_packet

        if not quiet:
            print("Done reading file. Time to read file: ", time.time() - read_start_time, "seconds.")

    def parse_header_packet(self, packet):

        def parse_metadata(data):
            # Metadata represented as a dictionary
            metadata = {}

            # Shorthand name expansions
            shorthand = {
                "_c": "study_centre",
                "_s": "study_code",
                "_i": "investigator",
                "_x": "exercise_code",
                "_v": "volunteer_num",
                "_p": "body_location",
                "_so": "setup_operator",
                "_n": "notes",
                "_b": "start_time",
                "_e": "end_time",
                "_ro": "recovery_operator",
                "_r": "retrieval_time",
                "_co": "comments",
                "_sc": "subject_code",
                "_se": "sex",
                "_h": "height",
                "_w": "weight",
                "_ha": "handedness",
            }

            # CWA File has 448 bytes of metadata at offset 64
            if sys.version_info[0] < 3:
                enc_string = str(data)
            else:
                enc_string = str(data, 'ascii')

            # Remove any trailing spaces, null, or 0xFF bytes
            enc_string = enc_string.rstrip('\x20\xff\x00')

            # Name-value pairs separated with ampersand
            name_values = enc_string.split('&')

            # Each name-value pair separated with an equals
            for name_value in name_values:
                parts = name_value.split('=')
                # Name is URL-encoded UTF-8
                name = urldecode(parts[0])
                if len(name) > 0:
                    value = None

                    if len(parts) > 1:
                        # Value is URL-encoded UTF-8
                        value = urldecode(parts[1])

                    # Expand shorthand names
                    name = shorthand.get(name, name)

                    # Store metadata name-value pair
                    metadata[name] = value

            # Metadata dictionary
            return metadata

        header = {}

        packet_header = packet[0:2].decode()                             # @ 0  +2   ASCII "MD", little-endian (0x444D)
        packet_length = unpack('<H', packet[2:4])[0]                     # @ 2  +2   Packet length (1020 bytes, with header (4) = 1024 bytes total)

        # TODO: message if packet header not MD
        if packet_header == 'MD' and packet_length == 1020:

            header['packet_length'] = packet_length

            hardware_type = unpack('B', packet[4:5])[0]                  # @ 4  +1   Hardware type (0x00/0xff/0x17 = AX3, 0x64 = AX6)
            header['hardware_type'] = hardware_type
            if hardware_type in [0x00, 0xff, 0x17]:
                header['device_type'] = 'AX3'
            elif hardware_type == 0x64:
                header['device_type'] = 'AX6'
            else:
                header['device_type'] = hex(hardware_type)[2:]

            header['device_id'] = unpack('<H', packet[5:7])[0]           # @ 5  +2   Device identifier
            header['session_id'] = unpack('<I', packet[7:11])[0]         # @ 7  +4   Unique session identifier

            device_id_upper = unpack('<H', packet[11:13])[0]             # @11  +2   Upper word of device id (if 0xffff is read, treat as 0x0000)
            if device_id_upper != 0xffff:
                header['device_id'] |= device_id_upper << 16

            header['logging_start'] = read_timestamp(packet[13:17])      # @13  +4   Start time for delayed logging
            header['logging_end'] = read_timestamp(packet[17:21])        # @17  +4   Stop time for delayed logging
            header['logging_capacity'] = unpack('<I', packet[21:25])[0]  # @21  +4   (Deprecated: preset maximum number of samples to collect, 0 = unlimited)
            # header['reserved1'] = packet[25:26]						# @25  +1   (1 byte reserved)

            header['flash_led'] = unpack('B', packet[35:36])[0]          # @26  +1   Flash LED during recording
            if header['flash_led'] == 0xff:
                header['flash_led'] = 0

            # header['reserved2'] = packet[27:35]						# @25  +8   (8 bytes reserved)
            sensor_config = unpack('B', packet[35:36])[0]                # @35  +1   Fixed rate sensor configuration, 0x00 or 0xff means accel only, otherwise bottom nibble is gyro range (8000/2^n dps): 2=2000, 3=1000, 4=500, 5=250, 6=125, top nibble non-zero is magnetometer enabled.
            rate_code = unpack('B', packet[36:37])[0]                    # @36  +1   Sampling rate code, frequency (3200/(1<<(15-(rate & 0x0f)))) Hz, range (+/-g) (16 >> (rate >> 6)).
            header['sample_rate'] = (3200 / (1 << (15 - (rate_code & 0x0f))))
            header['accel_range'] = (16 >> (rate_code >> 6))
            if sensor_config != 0x00 and sensor_config != 0xff:
                header['gyro_range'] = 8000 / 2 ** (sensor_config & 0x0f)
            else:
                header['gyro_range'] = 0

            header['last_change_time'] = read_timestamp(packet[37:41])   # @37  +4   Last change metadata time
            header['firmware_revision'] = unpack('B', packet[41:42])[0]  # @41  +1   Firmware revision number
            # header['timeZone'] = unpack('<H', packet[42:44])[0]		# @42  +2   (Unused: originally reserved for a "Time Zone offset from UTC in minutes", 0xffff = -1 = unknown)
            # header['reserved3'] = packet[44:64]						# @44  +20  (20 bytes reserved)
            header['metadata'] = parse_metadata(packet[64:512])    # @64  +448 "Annotation" meta-data (448 ASCII characters, ignore trailing 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value pairs)
            # header['reserved'] = packet[512:1024]						# @512 +512 Reserved for device-specific meta-data (512 bytes, ASCII characters, ignore trailing 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value pairs, leading '&' if present?)
            # Parse rateCode

        return header

    def parse_data_header(self, packet):

        start_time = None
        num_axes = None
        sample_count = None
        packing_format = None
        accel_scale = None
        gyro_scale = None
        light = None
        temperature = None
        battery = None

        packet_header = packet[0:2].decode()  # @ 0  +2   ASCII "AX", little-endian (0x5841)
        packet_length = unpack('<H', packet[2:4])[0]  # @ 2  +2   Packet length (508 bytes, with header (4) = 512 bytes total)

        if packet_header == 'AX':

            device_fractional = unpack('<H', packet[4:6])[
                0]  # @ 4  +2   Top bit set: 15-bit fraction of a second for the time stamp, the timestampOffset was already adjusted to minimize this assuming ideal sample rate; Top bit clear: 15-bit device identifier, 0 = unknown;
            # session_id = unpack('<I', packet[6:10])[0]           # @ 6  +4   Unique session identifier, 0 = unknown
            # sequence_id = unpack('<I', packet[10:14])[0]         # @10  +4   Sequence counter (0-indexed), each packet has a new number (reset if restarted)
            timestamp = read_timestamp(packet[14:18])  # @14  +4   Last reported RTC value, 0 = unknown
            light_scale = unpack('<H', packet[18:20])[
                0]  # @18  +2   AAAGGGLLLLLLLLLL Bottom 10 bits is last recorded light sensor value in raw units, 0 = none; top three bits are unpacked accel scale (1/2^(8+n) g); next three bits are gyro scale (8000/2^n dps)
            temperature = unpack('<H', packet[20:22])[
                0]  # @20  +2   Last recorded temperature sensor value in raw units (bottom-10 bits), 0 = none; (top 6-bits reserved)
            # events = unpack('B', packet[22:23])[0]               # @22  +1   Event flags since last packet, b0 = resume logging, b1 = reserved for single-tap event, b2 = reserved for double-tap event, b3 = reserved, b4 = reserved for diagnostic hardware buffer, b5 = reserved for diagnostic software buffer, b6 = reserved for diagnostic internal flag, b7 = reserved)
            battery = unpack('B', packet[23:24])[0]  # @23  +1   Last recorded battery level in raw units, 0 = unknown
            rate_code = unpack('B', packet[24:25])[
                0]  # @24  +1   Sample rate code, frequency (3200/(1<<(15-(rate & 0x0f)))) Hz, range (+/-g) (16 >> (rate >> 6)).
            num_axes_bps = unpack('B', packet[25:26])[
                0]  # @25  +1   0x32 (top nibble: number of axes, 3=Axyz, 6=Gxyz/Axyz, 9=Gxyz/Axyz/Mxyz; bottom nibble: packing format - 2 = 3x 16-bit signed, 0 = 3x 10-bit signed + 2-bit exponent)
            timestamp_offset = unpack('<h', packet[26:28])[
                0]  # @26  +2   Relative sample index from the start of the buffer where the whole-second timestamp is valid
            sample_count = unpack('<H', packet[28:30])[
                0]  # @28  +2   Number of sensor samples (if this sector is full -- Axyz: 80 or 120 samples, Gxyz/Axyz: 40 samples)
            # raw_sample_data = packet[30:510]				# @30  +480 Raw sample data.  Each sample is either 3x/6x/9x 16-bit signed values (x, y, z) or one 32-bit packed value (The bits in bytes [3][2][1][0]: eezzzzzz zzzzyyyy yyyyyyxx xxxxxxxx, e = binary exponent, lsb on right)
            # checksum = unpack('<H', packet[510:512])[0]		# @510 +2   Checksum of packet (16-bit word-wise sum of the whole packet should be zero)

            # get frequency
            frequency = 3200 / (1 << (15 - (rate_code & 0x0f)))

            # # adjust timestamp_offset
            # time_fractional = 0
            # # if top-bit set, we have a fractional date
            # if device_fractional & 0x8000:
            #     # Need to undo backwards-compatible shim by calculating how many whole samples the fractional part of timestamp accounts for.
            #     time_fractional = (device_fractional & 0x7fff)# << 1  # use original deviceId field bottom 15-bits as 16-bit fractional time
            #     timestamp_offset = sample_count - (time_fractional * int(frequency)) / 32767  # undo the backwards-compatible shift (as we have a true fractional)
            #
            # # Calculate start_time
            # start_time = timestamp - timedelta(seconds=(timestamp_offset / int(frequency)))

            # it appears the both OMGUI and other openmovement coe samples adjust the timestamp to sample 0 rather than
            # sample 1 - this code corrects that and adjusts to sample 1 as it should whether using fractional or offset

            if device_fractional & 0x8000:

                # if top bit of device_fractional set calculate start_time using decimal time_stamp of last sample
                time_fractional = (device_fractional & 0x7fff) / 32767
                timestamp = timestamp + timedelta(seconds=time_fractional)
                start_time = timestamp - timedelta(seconds=(sample_count - 1) / int(frequency))

            else:
                # if top bit of device_fractional not set calculate start_time using timestamp and timestamp_offset
                start_time = timestamp - timedelta(seconds=(timestamp_offset - 1) / int(frequency))

            # calculate bytes per sample
            num_axes = (num_axes_bps >> 4) & 0x0f
            packing_format = num_axes_bps & 0x0f

            # bps = 4
            # if packing_format == 0:
            #     bps = (num_axes / 3) * 4
            # elif packing_format == 2:
            #     bps = packing_format * num_axes
            # packet_samples = 480 // bps
            # accel_range = 16 >> (rate_code >> 6)

            accel_scale = (light_scale >> 13) & 0x07
            accel_scale = 1 / (2 ** (8 + accel_scale))

            gyro_range = (light_scale >> 10) & 0x07
            gyro_range = 4000 / (2 ** gyro_range) * 2
            gyro_scale = gyro_range / (2 ** 15)

            # light
            light = light_scale & 0x3ff

            # temperature
            temperature = (temperature & 0x3ff) * 75.0 / 256 - 50  # conversion used according to openmovement

            # battery
            battery = (battery + 512.0) * 6000 / 1024 / 1000.0



        return start_time, num_axes, sample_count, packing_format, accel_scale, gyro_scale, light, temperature, battery

    def write(self, file_type='edf', out_file='', edf_header={}, edf_signal_headers=[], deid=False, quiet=False):

        if file_type == 'edf':

            # CHECK THAT FILE EXTENSION MATCHES fule_type??
            # DOES folder need to exist?

            if out_file == '':
                out_file = os.path.join(self.file_dir, self.file_name[:-3] + 'edf')

            if not quiet:
                print("Writing %s ..." % out_file)

            device_type = self.header['device_type']

            if device_type == 'AX6':
                device_type = 'AXV6'
            else:
                print(f'{device_type} not currently supported.')
                return False

            # find amount to trim from start to shift to next second integer (EDF files don't store start time to ms)
            trim_microseconds = (1000000 - self.header['start_time'].microsecond
                                 if self.header['start_time'].microsecond > 0
                                 else 0)
            trim_samples = round(self.header['sample_rate'] * (trim_microseconds / 1000000))
            trim_packets = round(self.header['packet_rate'] * (trim_microseconds / 1000000))

            meta_keys = self.header['metadata'].keys()

            subject_code = self.header['metadata']['subject_code'] if 'subject_code' in meta_keys else ''
            study_code = self.header['metadata']['study_code'] if 'study_code' in meta_keys else ''
            body_location = self.header['metadata']['body_location'].replace(' ', '_') \
                if 'body_location' in meta_keys else ''
            sex = self.header['metadata']['sex'] if 'sex' in meta_keys else ''

            # update header to match nwdata format
            header = {'patientcode': subject_code,
                      'gender': sex,
                      'birthdate': '',
                      'patientname': '',
                      'patient_additional': '',
                      'startdate': self.header['start_time'] + timedelta(microseconds=trim_microseconds),
                      'admincode': study_code,
                      'technician': '',
                      'equipment': device_type + '_' + str(self.header['device_id']),
                      'recording_additional': body_location}

            header.update(edf_header)

            if deid:
                header.update({
                    'gender': '',
                    'birthdate': ''})

            signal_headers = [{'label': "Gyroscope x",
                                    'transducer': "MEMS",
                                    'dimension': 'degree/s',
                                    'sample_rate': self.header['sample_rate'],
                                    'physical_max': max(self.data['gx']),
                                    'physical_min': min(self.data['gx']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''},
                                   {'label': "Gyroscope y",
                                    'transducer': "MEMS",
                                    'dimension': 'degree/s',
                                    'sample_rate': self.header['sample_rate'],
                                    'physical_max': max(self.data['gy']),
                                    'physical_min': min(self.data['gy']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''},
                                   {'label': "Gyroscope z",
                                    'transducer': "MEMS",
                                    'dimension': 'degree/s',
                                    'sample_rate': self.header['sample_rate'],
                                    'physical_max': max(self.data['gz']),
                                    'physical_min': min(self.data['gz']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''},
                                   {'label': "Accelerometer x",
                                    'transducer': "MEMS",
                                    'dimension': 'g',
                                    'sample_rate': self.header['sample_rate'],
                                    'physical_max': max(self.data['ax']),
                                    'physical_min': min(self.data['ax']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''},
                                   {'label': "Accelerometer y",
                                    'transducer': "MEMS",
                                    'dimension': 'g',
                                    'sample_rate': self.header['sample_rate'],
                                    'physical_max': max(self.data['ay']),
                                    'physical_min': min(self.data['ay']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''},
                                   {'label': "Accelerometer z",
                                    'transducer': "MEMS",
                                    'dimension': 'g',
                                    'sample_rate': self.header['sample_rate'],
                                    'physical_max': max(self.data['az']),
                                    'physical_min': min(self.data['az']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''},
                                   {'label': "Light",
                                    'transducer': "Logarithmic light sensor",
                                    'dimension': 'units',
                                    'sample_rate': self.header['packet_rate'],
                                    'physical_max': max(self.data['light']),
                                    'physical_min': min(self.data['light']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''},
                                   {'label': "Temperature",
                                    'transducer': "Linear thermistor",
                                    'dimension': 'C',
                                    'sample_rate': self.header['packet_rate'],
                                    'physical_max': max(self.data['temperature']),
                                    'physical_min': min(self.data['temperature']),
                                    'digital_max': 32767,
                                    'digital_min': -32768,
                                    'prefilter': ''}]

            sh_index = 0

            for signal_header in edf_signal_headers:
                signal_headers[sh_index].update(signal_header)
                sh_index = sh_index + 1

            # write to edf
            edf_file = EDFFile(out_file)
            edf_file.header = header
            edf_file.signal_headers = signal_headers
            edf_file.signals = [self.data['gx'][trim_samples:],
                                self.data['gy'][trim_samples:],
                                self.data['gz'][trim_samples:],
                                self.data['ax'][trim_samples:],
                                self.data['ay'][trim_samples:],
                                self.data['az'][trim_samples:],
                                self.data['light'][trim_packets:],
                                self.data['temperature'][trim_packets:]]
            edf_file.write(out_file, quiet=quiet)

            return True


def read_timestamp(data):
    value = unpack('<I', data)[0]
    if value == 0x00000000:  # Infinitely in past = 'always before now'
        return 0
    if value == 0xffffffff:  # Infinitely in future = 'always after now'
        return -1
    # bit pattern:  YYYYYYMM MMDDDDDh hhhhmmmm mmssssss
    year = ((value >> 26) & 0x3f) + 2000
    month = (value >> 22) & 0x0f
    day = (value >> 17) & 0x1f
    hours = (value >> 12) & 0x1f
    mins = (value >> 6) & 0x3f
    secs = (value >> 0) & 0x3f
    try:
        dt = datetime(year, month, day, hours, mins, secs)
        # timestamp = (dt - datetime(1970, 1, 1)).total_seconds()
        return dt
    # return str(datetime.fromtimestamp(timestamp))
    # return time.strptime(t, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        print("WARNING: Invalid date:", year, month, day, hours, mins, secs)
        return -1


# 16-bit checksum (should sum to zero)
def checksum(data):
    sum = 0
    for i in range(0, len(data), 2):
        # value = data[i] | (data[i + 1] << 8)
        value = unpack('<H', data[i:i + 2])[0]
        sum = (sum + value) & 0xffff
    return sum


# Local "URL-decode as UTF-8 string" function
def urldecode(input):
    output = bytearray()
    nibbles = 0
    value = 0
    # Each input character
    for char in input:
        if char == '%':
            # Begin a percent-encoded hex pair
            nibbles = 2
            value = 0
        elif nibbles > 0:
            # Parse the percent-encoded hex digits
            value *= 16
            if char >= 'a' and char <= 'f':
                value += ord(char) + 10 - ord('a')
            elif char >= 'A' and char <= 'F':
                value += ord(char) + 10 - ord('A')
            elif char >= '0' and char <= '9':
                value += ord(char) - ord('0')
            nibbles -= 1
            if nibbles == 0:
                output.append(value)
        elif char == '+':
            # Treat plus as space (application/x-www-form-urlencoded)
            output.append(ord(' '))
        else:
            # Preserve character
            output.append(ord(char))

    return output.decode('utf-8')

if __name__ == '__main__':

    cwa_path = r'W:\NiMBaLWEAR\OND09\wearables\raw\OND09_0001_01_AXV6_RAnkle.cwa'

    device = CWAFile(cwa_path)
    device.read_header()