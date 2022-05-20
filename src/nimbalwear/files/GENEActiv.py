# Authors:  David Ding
#           Kit Beyer
#           Adam Vert

# ======================================== IMPORTS ========================================
import numpy as np
import datetime
import os
import shutil
import fpdf
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import time
from tqdm import tqdm

from src.nimbalwear.files import EDF

mstyle.use('fast')


# ======================================== GENEActivFile CLASS ========================================
class GENEActivFile:

    def __init__(self, file_path):

        self.file_path = os.path.abspath(file_path)
        self.file_name = os.path.basename(self.file_path)
        self.file_dir = os.path.dirname(self.file_path)
        self.header = {}
        self.data_packet = None

        self.pagecount = None
        self.pagecount_match = None
        self.samples = None
        self.clock_drift_rate = None

        # data read from file (may be partial pages and/or downsampled - see file_metadata)
        self.data = {
            'x': None,
            'y': None,
            'z': None,
            'temperature': None,
            'light': None,
            'button': None,
            'start_page': None,
            'end_page': None,
            'start_time': None,
            'sample_rate': None,
            'temperature_sample_rate': None}

    def read(self, parse_data=True, start=1, end=-1, downsample=1, calibrate=True, correct_drift=False, quiet=False):

        """
        read() reads a raw GENEActiv .bin file
        Args:
            parse_data: bool
                Parse hexadecimal data
            start: int

            end: int

            downsample: int

            calibrate: bool

            correct_drift: bool

            update: bool

            quiet: bool
                Suppress text feedback

        Returns:

        """

        read_start_time = time.time()

        # if file does not exist then exit
        if not os.path.exists(self.file_path):
            print(f"****** WARNING: {self.file_path} does not exist.\n")
            return

        # Read GENEActiv .bin file
        if not quiet:
            print("Reading %s ..." % self.file_path)

        bin_file = open(self.file_path, 'r', encoding='utf-8')
        lines = [line[:-1] for line in bin_file.readlines()]
        bin_file.close()

        # Calculate number of lines in header
        header_end = lines[:150].index("Recorded Data")

        # Separate header and data packets
        header_packet = lines[:header_end]
        self.data_packet = lines[header_end:]

        # Parse header into header dict
        if not quiet:
            print("Parsing header information ...")

        for line in header_packet:
            try:
                colon = line.index(':')
                self.header[line[:colon]] = line[colon + 1:].rstrip('\x00').rstrip()
            except ValueError:
                pass

        # calculate pagecount from data_packet

        # set match to true
        self.pagecount_match = True

        # get page counts
        self.pagecount = len(self.data_packet) / 10
        header_pagecount = int(self.header['Number of Pages'])

        # check if pages read is an integer (lines read is multiple of 10)
        if not self.pagecount.is_integer():
            # set match to false and display warning
            self.pagecount_match = False
            print(f"****** WARNING: Pages read ({self.pagecount}) is not",
                  f"an integer, data may be corrupt.\n")

        # check if pages read matches header count
        if self.pagecount != header_pagecount:
            # set match to false and display warning
            self.pagecount_match = False
            print(f"****** WARNING: Pages read ({self.pagecount}) not equal to",
                  f"'Number of Pages' in header ({header_pagecount}).\n")

        # cacluate number of samples
        self.samples = self.pagecount * 300

        # calculate clock drift rate
        config_time = datetime.datetime.strptime(self.header['Config Time'], '%Y-%m-%d %H:%M:%S:%f')
        extract_time = datetime.datetime.strptime(self.header['Extract Time'], '%Y-%m-%d %H:%M:%S:%f')
        clock_drift = float(self.header['Extract Notes'].split(' ')[3][:-2].replace(',', ''))

        if clock_drift > 3000:
            clock_drift = clock_drift - 3600
        elif clock_drift < -3000:
            clock_drift = clock_drift + 3600

        total_seconds = (extract_time - config_time).total_seconds()
        self.clock_drift_rate = clock_drift / total_seconds

        # parse data from hexadecimal
        if parse_data:
            self.parse_data(start=start, end=end, downsample=downsample, calibrate=calibrate,
                            correct_drift=correct_drift, quiet=quiet)

        if not quiet:
            print("Done reading file. Time to read file: ", time.time() - read_start_time, "seconds.")

    def parse_data(self, start=1, end=-1, downsample=1, calibrate=True, correct_drift=False, quiet=False):

        def twos_comp(val, bits):
            """ This method calculates the twos complement value of the current bit
            Args:
                val: bin
                    Bits to be processed (Binary)
                bits: int
                    Total number of bits in the operation

            Returns:
                page_data: Integer value resulting from the twos compliment operation
            """
            if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
                val = val - (1 << bits)  # compute negative value
            return val

        # check whether data has been read
        if not self.header or self.data_packet is None or self.pagecount is None:
            print("****** WARNING: Cannot parse data because file has not been read.\n")
            return

        if not quiet:
            print("Parsing data from hexadecimal ...")

        # store passed arguments before checking and modifying
        old_start = start
        old_end = end
        old_downsample = downsample

        # check start and end for acceptable values
        if start < 1:
            start = 1
        elif start > self.pagecount:
            start = round(self.pagecount)

        if end == -1 or end > self.pagecount:
            end = round(self.pagecount)
        elif end < start:
            end = start

        # check downsample for valid values
        if downsample < 1:
            downsample = 1
        elif downsample > 6:
            downsample = 6

        self.data = {
            'x': [],
            'y': [],
            'z': [],
            'temperature': [],
            'light': [],
            'button': [],
            'start_page': None,
            'end_page': None,
            'start_time': None,
            'sample_rate': None,
            'temperature_sample_rate': None}

        # get calibration variables
        if calibrate:
            x_offset = int(self.header['x offset'])
            y_offset = int(self.header['y offset'])
            z_offset = int(self.header['z offset'])
            x_gain = int(self.header['x gain'])
            y_gain = int(self.header['y gain'])
            z_gain = int(self.header['z gain'])
            lux = int(self.header['Lux'])
            volts = int(self.header['Volts'])
        else:
            x_offset = 0
            y_offset = 0
            z_offset = 0
            x_gain = 1
            y_gain = 1
            z_gain = 1
            lux = 0
            volts = 1

        sample_rate = int(self.header['Measurement Frequency'].split(' ')[0])
        downsampled_rate = (sample_rate / downsample)

        # get start_time (time of first data point in view)
        start_time_line = self.data_packet[(start - 1) * 10 + 3]
        colon = start_time_line.index(':')
        start_time = start_time_line[colon + 1:]
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S:%f')

        # grab chunk of data from packet
        data_chunk = [self.data_packet[i]
                      for i in range((start - 1) * 10 + 9, end * 10, 10)]

        i = 0

        # loop through pages
        for data_line in tqdm(data_chunk, leave=False, desc='Parsing bin data'):

            i += 1

            # loop through measurements in page
            for j in range(0, 300, downsample):

                # parse measurement from line and convert from hex to bin
                meas = data_line[j * 12 : (j + 1) * 12]
                meas = bin(int(meas, 16))[2:]
                meas = meas.zfill(48)

                # parse each signal from measurement and convert to int
                meas_x = int(meas[0:12], 2)
                meas_y = int(meas[12:24], 2)
                meas_z = int(meas[24:36], 2)
                meas_light = int(meas[36:46], 2)
                meas_button = int(meas[46], 2)

                # use twos complement to get signed integer for accelerometer data
                meas_x = twos_comp(meas_x, 12)
                meas_y = twos_comp(meas_y, 12)
                meas_z = twos_comp(meas_z, 12)

                # calibrate data if requested
                if calibrate:
                    meas_x = (meas_x * 100 - x_offset) / x_gain
                    meas_y = (meas_y * 100 - y_offset) / y_gain
                    meas_z = (meas_z * 100 - z_offset) / z_gain
                    meas_light = (meas_light * lux) / volts

                # append measurement to data list
                self.data['x'].append(meas_x)
                self.data['y'].append(meas_y)
                self.data['z'].append(meas_z)
                self.data['light'].append(meas_light)
                self.data['button'].append(meas_button)

        # get all temperature lines from data packet (1 per page)
        temperature_chunk = [self.data_packet[i]
                             for i in range((start - 1) * 10 + 5, end * 10, 10)]

        # parse temperature from temperature lines and insert into dict
        for temperature_line in tqdm(temperature_chunk, leave=False, desc='Parsing temperature data'):
            colon = temperature_line.index(':')
            self.data['temperature'].append(float(temperature_line[colon + 1:]))

        self.data['start_page'] = start
        self.data['end_page'] = end
        self.data['start_time'] = start_time
        self.data['sample_rate'] = downsampled_rate
        self.data['temperature_sample_rate'] = sample_rate / 300

        signal_keys = ['x', 'y', 'z', 'light', 'button', 'temperature']

        # correct clock drift
        if correct_drift:

            if not quiet:
                print("Correcting clock drift ...")

            config_time = datetime.datetime.strptime(self.header['Config Time'], '%Y-%m-%d %H:%M:%S:%f')
            adjust_rate = abs(1 / self.clock_drift_rate)
            time_to_start = (self.data['start_time'] - config_time).total_seconds()
            adjust_start = int(time_to_start * self.data['sample_rate'] * abs(self.clock_drift_rate))
            adjust_start_temperature = \
                int(time_to_start * self.data['temperature_sample_rate'] * abs(self.clock_drift_rate))

            if self.clock_drift_rate > 0:  # if drift is positive then remove extra samples

                for key in tqdm(signal_keys, leave=False, desc='Correcting clock drift (delete)'):

                    # delete data from each signal
                    self.data[key] = np.delete(self.data[key],
                                          [round(adjust_rate * (i + 1)) for i in
                                           range(int(len(self.data[key]) / adjust_rate))])

                    # delete data from start of each signal to account for time from config to start
                    if key == 'temperature':
                        self.data[key] = np.delete(self.data[key], range(adjust_start_temperature))
                    else:
                        self.data[key] = np.delete(self.data[key], range(adjust_start))

                    self.data[key] = self.data[key]

            else:  # else add samples

                for key in tqdm(signal_keys, leave=False, desc='Correcting clock drift (insert)'):

                    insert_count = int(len(self.data[key]) / adjust_rate)
                    insert_before = [round(adjust_rate * i) for i in range(1, insert_count)]
                    insert_value = [(self.data[key][i-1] + self.data[key][i]) / 2 for i in insert_before]

                    # insert data into each signal
                    self.data[key] = np.insert(self.data[key], insert_before, insert_value)

                    # insert data into start of each signal to account for time from config to start
                    if key == 'temperature':
                        self.data[key] = np.insert(self.data[key], 0, [0] * adjust_start_temperature)
                    else:
                        self.data[key] = np.insert(self.data[key], 0, [0] * adjust_start)

                    self.data[key] = self.data[key]

        else:

            for key in tqdm(signal_keys, leave=False, desc='Converting to ndarray'):

                # convert to ndarray
                self.data[key] = np.array(self.data[key])

        # display message if start values were changed
        if old_start != start:
            print("****** WARNING: Start or end values were modified to fit acceptable range.\n",
                  f"       Old range: {old_start} to {old_end}\n",
                  f"       New range: {start} to {end}\n")

        # display message if end values were changed
        if old_end not in [end, -1]:
            print("****** WARNING: Start or end values were modified to fit acceptable range.\n",
                  f"       Old range: {old_start} to {old_end}\n",
                  f"       New range: {start} to {end}\n")

        # display message downsample ratio was changed
        if old_downsample != downsample:
            print("****** WARNING: Downsample value was modified to fit acceptable range.\n",
                  f"       Old value: {old_downsample}\n",
                  f"       New value: {downsample}\n")

        return True

    def create_pdf(self, pdf_folder, window_hours=4, downsample=5, correct_drift=False, quiet=False):

        """WILL NOT WORK SINCE OPTION TO UPDATE DATA WAS REMOVED - NEEDS TO BE FIXED
        creates a pdf summary of the file
        Parameters
        ----------
        pdf_folder : str
            path to folder where pdf will be stored
        window_hours : int
            number of hours to display on each page (default = 4) -- if hour occurs
            in the middle of a data page then time displayed on each pdf page may
            be slightly less than the number of hours specified
        downsample : int
            factor by which to downsample (range: 1-6, default = 5)
        correct_drift: bool
            should sample rate be adjusted for clock drift? (default = False)
        quiet: bool
            suppress text messages (default = False)
        
        Returns
        -------
        pdf_path : str
            path to pdf file created
        """

        # check whether data has been read
        if not self.header or self.data_packet is None or self.pagecount is None:
            print("****** WARNING: Cannot view data because file has not",
                  "been read.")
            return

        if not quiet:
            print("Creating PDF summary ...")

        # get filenames and paths

        base_name = os.path.splitext(self.file_name)[0]

        pdf_name = base_name + '.pdf'
        pdf_path = os.path.join(pdf_folder, pdf_name)

        png_folder = os.path.join(pdf_folder, 'png', '')

        # adjust sample rate for clock drift?
        sample_rate = int(self.header["Measurement Frequency"].split(" ")[0])

        # calculate pages per plot
        window_pages = round((window_hours * 60 * 60 * sample_rate) / 300)
        window_sequence = range(1, round(self.pagecount), window_pages)
        # window_sequence = range(1, window_pages*6, window_pages)

        # CREATE PLOTS ------

        if not quiet:
            print("Generating plots ...")

        # define date locators and formatters
        # hours = mdates.HourLocator()
        # hours_fmt = mdates.DateFormatter('%H:%M')

        # set plot parameters

        # each accelerometer axis has a different min and max based on the digital range
        # and the offset and gain values (-8 to 8 stated in the header is just
        # a minimum range, actual range is slightly larger)

        x_min = (-204800 - int(self.header['x offset'])) / int(self.header['x gain'])
        y_min = (-204800 - int(self.header['y offset'])) / int(self.header['y gain'])
        z_min = (-204800 - int(self.header['z offset'])) / int(self.header['z gain'])
        x_max = (204700 - int(self.header['x offset'])) / int(self.header['x gain'])
        y_max = (204700 - int(self.header['y offset'])) / int(self.header['y gain'])
        z_max = (204700 - int(self.header['z offset'])) / int(self.header['z gain'])
        light_min = 0 * int(self.header['Lux']) / int(self.header['Volts'])
        light_max = 1023 * int(self.header['Lux']) / int(self.header['Volts'])

        accelerometer_min = min([x_min, y_min, z_min])
        accelerometer_max = max([x_max, y_max, z_max])
        accelerometer_range = accelerometer_max - accelerometer_min
        accelerometer_buffer = accelerometer_range * 0.1

        light_range = light_max - light_min
        light_buffer = light_range * 0.1

        yaxis_lim = [[accelerometer_min - accelerometer_buffer, accelerometer_max + accelerometer_buffer],
                     [accelerometer_min - accelerometer_buffer, accelerometer_max + accelerometer_buffer],
                     [accelerometer_min - accelerometer_buffer, accelerometer_max + accelerometer_buffer],
                     [light_min - light_buffer, light_max + light_buffer],
                     [-0.01, 1],
                     [9.99, 40.01]]

        yaxis_ticks = [[-8, 0, 8],
                       [-8, 0, 8],
                       [-8, 0, 8],
                       [0, 10000, 20000, 30000],
                       [0, 1],
                       [10, 20, 30, 40]]

        yaxis_units = [self.header["Accelerometer Units"],
                       self.header["Accelerometer Units"],
                       self.header["Accelerometer Units"],
                       self.header["Light Meter Units"],
                       "",
                       self.header["Temperature Sensor Units"]]

        yaxis_lines = [[x_min, 0, x_max],
                       [y_min, 0, y_max],
                       [z_min, 0, z_max],
                       [light_min, light_max]]

        line_color = ['b', 'g', 'r', 'c', 'm', 'y']

        plt.rcParams['lines.linewidth'] = 0.25
        plt.rcParams['figure.figsize'] = (6, 7.5)
        plt.rcParams['figure.subplot.top'] = 0.92
        plt.rcParams['figure.subplot.bottom'] = 0.06
        plt.rcParams['font.size'] = 8

        # create temperature folder to store .png files
        if not os.path.exists(png_folder):
            os.mkdir(png_folder)

        # loop through time windows to create separate plot for each
        for start_index in window_sequence:

            # get data for current window
            end_index = start_index + window_pages - 1
            plot_data = self.parse_data(start=start_index,
                                        end=end_index,
                                        downsample=downsample,
                                        correct_drift=correct_drift,
                                        quiet=quiet)

            # format start and end date for current window
            time_format = '%b %d, %Y (%A) @ %H:%M:%S.%f'
            window_start = plot_data['start_time']
            window_start_txt = window_start.strftime(time_format)[:-3]

            window_end = window_start + datetime.timedelta(hours=window_hours)
            window_end_txt = window_end.strftime(time_format)[:-3]

            # initialize figure with subplots
            fig, ax = plt.subplots(6, 1)

            # insert date range as plot title
            fig.suptitle(f"{window_start_txt} to {window_end_txt}",
                         fontsize=8, y=0.96)

            # initialize subplot index
            subplot_index = 0

            # loop through subplots and generate plot
            for key in ['x', 'y', 'z', 'light', 'button', 'temperature']:

                # plot signal
                ax[subplot_index].plot(plot_data[key],
                                       color=line_color[subplot_index])

                # remove box around plot
                ax[subplot_index].spines['top'].set_visible(False)
                ax[subplot_index].spines['bottom'].set_visible(False)
                ax[subplot_index].spines['right'].set_visible(False)

                #                # set axis ticks and labels
                #                ax[subplot_index].xaxis.set_major_locator(hours)
                #                ax[subplot_index].xaxis.set_major_formatter(hours_fmt)
                #                if subplot_index != 5:
                #                    ax[subplot_index].set_xticklabels([])

                ax[subplot_index].set_yticks(yaxis_ticks[subplot_index])
                units = yaxis_units[subplot_index]
                ax[subplot_index].set_ylabel(f"{key} ({units})")

                #                # set vertical lines on plot at hours
                #                ax[subplot_index].grid(True, 'major', 'x',
                #                                       color = 'k', linestyle = '--')

                # set horizontal lines on plot at zero and limits
                if subplot_index < 4:
                    for yline in yaxis_lines[subplot_index]:
                        ax[subplot_index].axhline(y=yline, color='grey', linestyle='-')

                # set axis limits
                ax[subplot_index].set_ylim(yaxis_lim[subplot_index])
                #                ax[subplot_index].set_xlim(window_start,
                #                                           window_start +
                #                                           dt.timedelta(hours = 4))

                # increment to next subplot
                subplot_index += 1

            # save figure as .png and close
            png_file = "plot_" + f"{start_index:09d}" + ".png"
            fig.savefig(os.path.join(png_folder, png_file))
            plt.close(fig)

        # CREATE PDF ------

        if not quiet:
            print("Building PDF ...")

        # HEADER PAGE ----------------

        # initialize pdf
        pdf = fpdf.FPDF(format='letter')

        # add first page and print file name at top
        pdf.add_page()
        pdf.set_font('Courier', size=16)
        pdf.cell(200, 10, txt=self.file_name, ln=1, align='C', border=0)

        # set font for header info
        pdf.set_font('Courier', size=12)
        header_text = '\n'

        # find length of longest key in header
        key_length = max(len(key) for key in self.header.keys()) + 1

        # create text string for header information
        for key, value in self.header.items():
            header_text = header_text + f"{key:{key_length}}:  {value}\n"

        # print header to pdf
        pdf.multi_cell(200, 5, txt=header_text, align='L')

        # PLOT DATA PAGES -------------

        # list all .png files in temperature folder
        png_files = os.listdir(png_folder)
        png_files.sort()

        # loop through .png files to add to pdf
        for png_file in png_files:
            # create full .png file path
            png_path = os.path.join(png_folder, png_file)

            # add page and set font
            pdf.add_page()
            pdf.set_font('Courier', size=16)

            # print file_name as header
            pdf.cell(0, txt=self.file_name, align='C')
            pdf.ln()

            # insert .png plot into pdf
            pdf.image(png_path, x=1, y=13, type='png')

        # SAVE PDF AND DELETE PNGS --------------

        if not quiet:
            print("Cleaning up ...")

        # save pdf file
        pdf.output(pdf_path)

        # delete temperature .png files
        shutil.rmtree(png_folder)

        if not quiet:
            print("Done creating PDF summary ...")

        return pdf_path

    def write(self, file_type='edf', out_file='', edf_header={}, edf_signal_headers=[], deid=False, quiet=False):

        # check whether data has been read

        # pass in custome header and signalheaders and update dict

        if file_type == 'edf':

            # CHECK THAT FILE EXTENSION MATCHES fule_type??
            # DOES folder need to exist?

            if out_file == '':
                out_file = os.path.join(self.file_dir, self.file_name[:-3] + 'edf')

            if not quiet:
                print("Writing %s ..." % out_file)

            #  Number of samples to remove to get as close as possible to the next second
            trim_microseconds = (1000000 - self.data['start_time'].microsecond
                                 if self.data['start_time'].microsecond > 0
                                 else 0)

            trim_samples = round(self.data['sample_rate'] * (trim_microseconds / 1000000))

            header = {'patientcode': self.header['Subject Code'],
                      'gender': self.header['Sex'],
                      'birthdate': datetime.datetime.strptime(self.header["Date of Birth"], "%Y-%m-%d"),
                      'patientname': '',
                      'patient_additional': '',
                      'startdate': self.data['start_time'] + datetime.timedelta(microseconds=trim_microseconds),
                      'admincode': self.header['Study Code'],
                      'technician': self.header['Investigator ID'],
                      'equipment': self.header['Device Type'] + '_' + self.header['Device Unique Serial Code'],
                      'recording_additional': self.header['Device Location Code'].replace(' ', '_')}

            header.update(edf_header)

            if deid:
                header.update({
                    'gender': '',
                    'birthdate': ''})

            signal_headers = [{'label': "Accelerometer x",
                               'transducer': "MEMS Accelerometer",
                               'dimension': self.header['Accelerometer Units'],
                               'sample_rate': 300,
                               'physical_max': (204700 - int(self.header['x offset'])) / int(self.header['x gain']),
                               'physical_min': (-204800 - int(self.header['x offset'])) / int(self.header['x gain']),
                               'digital_max': 32767,
                               'digital_min': -32768,
                               'prefilter': ''},
                              {'label': "Accelerometer y",
                               'transducer': "MEMS Accelerometer",
                               'dimension': self.header['Accelerometer Units'],
                               'sample_rate': 300,
                               'physical_max': (204700 - int(self.header['y offset'])) / int(self.header['y gain']),
                               'physical_min': (-204800 - int(self.header['y offset'])) / int(self.header['y gain']),
                               'digital_max': 32767,
                               'digital_min': -32768,
                               'prefilter': ''},
                              {'label': "Accelerometer z",
                               'transducer': "MEMS Accelerometer",
                               'dimension': self.header['Accelerometer Units'],
                               'sample_rate': 300,
                               'physical_max': (204700 - int(self.header['z offset'])) / int(self.header['z gain']),
                               'physical_min': (-204800 - int(self.header['z offset'])) / int(self.header['z gain']),
                               'digital_max': 32767,
                               'digital_min': -32768,
                               'prefilter': ''},
                              {'label': "Temperature",
                               'transducer': "Linear active thermistor",
                               'dimension': self.header['Temperature Sensor Units'],
                               'sample_rate': 1,
                               'physical_max': int(self.header["Temperature Sensor Range"][5:7]),
                               'physical_min': int(self.header["Temperature Sensor Range"][0]),
                               'digital_max': 32767,
                               'digital_min': -32768,
                               'prefilter': ''},
                              {'label': "Light",
                               'transducer': "Silicon photodiode",
                               'dimension': self.header['Light Meter Units'],
                               'sample_rate': 300,
                               'physical_max': 1023 * int(self.header['Lux']) / int(self.header['Volts']),
                               'physical_min': 0 * int(self.header['Lux']) / int(self.header['Volts']),
                               'digital_max': 32767,
                               'digital_min': -32768,
                               'prefilter': ''},
                              {'label': "Button",
                               'transducer': "Mechanical membrane switch",
                               'dimension': '',
                               'sample_rate': 300,
                               'physical_max': 1,
                               'physical_min': 0,
                               'digital_max': 32767,
                               'digital_min': -32768,
                               'prefilter': ''}]

            sh_index = 0

            for signal_header in edf_signal_headers:
                signal_headers[sh_index].update(signal_header)
                sh_index = sh_index + 1

            # write to edf
            edf_file = EDF.EDFFile(out_file)
            edf_file.header = header
            edf_file.signal_headers = signal_headers
            edf_file.signals = [self.data['x'][trim_samples:],
                                     self.data['y'][trim_samples:],
                                     self.data['z'][trim_samples:],
                                     self.data['temperature'],
                                     self.data['light'][trim_samples:],
                                     self.data['button'][trim_samples:]]
            edf_file.write(out_file, quiet=quiet)

            return True

def convert_gnac_dir_edf(in_path, out_path, correct_drift=False, deid=False, overwrite=False, quiet=True):

    #add option to not ovewrite

    file_list = [f for f in os.listdir(in_path)
                 if f.lower().endswith('.bin') and not f.startswith('.')]

    if not overwrite:
        file_list = [f for f in file_list if f.replace('.bin', '.edf') not in os.listdir(out_path)]

    file_list.sort()

    for file_name in tqdm(file_list):
        ga_file = GENEActivFile(os.path.join(in_path, file_name))
        ga_file.read(correct_drift=correct_drift, quiet=quiet)
        ga_file.write(out_file=os.path.join(out_path, file_name[:-3] + 'edf'), deid=deid, quiet=quiet)