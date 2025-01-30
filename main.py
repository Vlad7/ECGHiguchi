# This is an ECG Higuchi script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Age groups
import math
import pandas as pd
import openpyxl
import enum
#from IPython.display import display
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv
import matplotlib.pyplot as plt
import os.path
import sys
import os

#<<<<<<< HEAD
# Импортируем функцию bwr
#=======
#>>>>>>> fd1fd4467c513d087a5e66e3e79bb722d5280811


# Добавляем родительскую директорию
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py-bwr')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pan_tompkins')))

# Импортируем функцию bwr
import bwr

import bwr
#import nbimporter
import pan_tompkins

import neurokit2 as nk




gender = {'NaN': 'none', '0': 'male', '1': 'female'}
device = {'NaN': 'none', '0': 'TFM, CNSystems', '1': 'CNAP 500, CNSystems; MP150, BIOPAC Systems'}

age_groups = {'NaN': 'none',
              '1': '18 - 19',
                  '2': '20 - 24',
                  '3': '25 - 29',
                  '4': '30 - 34',
                  '5': '35 - 39',
                  '6': '40 - 44',
                  '7': '45 - 49',
                  '8': '50 - 54',
                  '9': '55 - 59',
                  '10': '60 - 64',
                  '11': '65 - 69',
                  '12': '70 - 74',
                  '13': '75 - 79',
                  '14': '80 - 84',
                  '15': '85 - 92',
                  }

class TypeOfECGCut(enum.Enum):

    full = 1
    start = 2
    middle = 3
    end = 4


#<<<<<<< HEAD
#=======
#from IPython.display import display
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
from biosppy import storage
from biosppy.signals import ecg
#>>>>>>> a7e22fc04678f933cea1483c155fdcd1e8416900
#######################################################################################################################

# Path to dataset of ECG
# For future make loading from web database
#path_to_dataset_folder = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'
path_to_dataset_folder  = 'C:/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'
csv_info_file = 'subject-info.csv'

rr_intervals_folder="rr_intervals/all"
#######################################################################################################################


minimum_length_of_ECG = 480501


##############################################################################################
################################## Parameters for ECG cutting ################################
##############################################################################################

# Total minutes that should to wait after activity before ECG mading
total_minutes_points_from_ECG_start = 5

# Expected minutes waited before ECG mading from selected database
expected_minutes_points_that_ECG_waited = 2

##############################################################################################
##############################################################################################
##############################################################################################

DATABASE_ATTRIBUTES = []
DATABASE = {}

def print_database_attributes():
    print(f'Column names are: {", ".join(DATABASE_ATTRIBUTES)} ')


class RECORD:

    """ Record class that represents necessary information about ECG database """

    # Database with filtered ages

    def __init__(self, id, age_group, sex, bmi, length, device, ecg_1, ecg_2):

        """Without third time series - pressure"""

        self.Id = id
        self.AgeGroup = age_group
        self.Sex = sex
        self.BMI = bmi
        self.Length = length
        self.Device = device
        #self.ECG_1 = ecg_1
        #self.ECG_2 = ecg_2

        DATABASE[id] = self


    def __str__(self):

        self.plot()
        return str(f'\tId: {self.Id:>6};    Age_group: {age_groups[self.AgeGroup]:>9};    Sex: {gender[self.Sex]:>6};    BMI: {self.BMI:>6};    Length: {self.Length:>5};    Device: {device[self.Device]}.')

    def plot(self):

        plt.plot(self.ECG_1)
        plt.ylabel("mV")
        plt.show()
        #wfdb.plot_wfdb(record, title='Record ' + id + ' from Physionet Autonomic ECG')
# ECG_dictionary with information about ECG



def print_database():
    for id in DATABASE.keys():
       print(DATABASE[id])



#######################################################################################################################
#######################################################################################################################
#######################################################################################################################













def calculate_higuchi(ECG_1, ECG_2=None, num_k_value=50, k_max_value=None):

    """For the case when two ECG.
        Input parameters:
        num_k_value - number of k values
        k_max_value - value of Kmax"""


    dictionary_HFD_ECG_1 = {}
    #dictionary_HFD_ECG_2 = {}
    
    dictionary_ages = {}



    for key in ECG_1.keys():
        HFD_1 = HiguchiFractalDimension.hfd(np.array(ECG_1[key]), opt=True, num_k=num_k_value,
                                          k_max=k_max_value)

        dictionary_HFD_ECG_1[key] = HFD_1

    print(dictionary_HFD_ECG_1)

    dictionary_ages = extract_ages(ECG_1.keys())
    print(dictionary_ages)

    dictionary_age_ranges = {}

    for key in dictionary_ages.keys():
        dictionary_age_ranges[key] = age_groups[dictionary_ages[key]]

    write_HFD_calculated_values_to_csv(dictionary_HFD_ECG_1,dictionary_ages,dictionary_age_ranges)

    age_category_ids_dictionary = {}

    # For each age range list of id's

    for key in dictionary_ages.keys():

        if age_category_ids_dictionary.keys().__contains__(dictionary_ages[key]):
            age_category_ids_dictionary[dictionary_ages[key]].append(key)
        else:
            age_category_ids_dictionary[dictionary_ages[key]] = [key]

    HFD_average_by_age_range = {}

    for key in age_category_ids_dictionary.keys():

        HFD_1_summ = 0
        #HFD_2_average = 0

        for age_range_key in age_category_ids_dictionary[key]:
            HFD_1_summ += dictionary_HFD_ECG_1[age_range_key]
            #HFD_2_average += dictionary_HFD_ECG_1_2[age_range_key][1]

        length_of_age_range_ids_list = len(age_category_ids_dictionary[key])
        HFD_average_by_age_range[key] = HFD_1_summ / length_of_age_range_ids_list

    write_average_HFD_values_for_each_age_range(HFD_average_by_age_range)

    #ECG_count_per_age_group_dictionary = {}

    #Higuchi_average_per_age_group_dictionary = {}




    #HFD_2 = HiguchiFractalDimension.hfd(np.array(ECG_2), opt=True, num_k=num_k_value, k_max=k_max_value)

    """
    if (not math.isnan(HFD_1)):
        dictionary_HFD_ECG_1[ecg.Id] = HFD_1

    if (not math.isnan(HFD_2)):
        dictionary_HFD_ECG_2[ecg.Id] = HFD_2"""

    """
    if ((not math.isnan(HFD_1)) and (not math.isnan(HFD_2))):
        return [HFD_1, HFD_2]
    else:
        return None
    """

    # For testing
    #dictionary_HFD_ECG_1.pop("0001")
    #dictionary_HFD_ECG_2.pop("0010")

    # Intersect of two sets
    #keys = list(set(dictionary_HFD_ECG_1.keys()) & set(dictionary_HFD_ECG_2.keys()))

    """dictionary_ages = {}
    for key in keys:
        dictionary_HFD_ECG_1_2[key] = [dictionary_HFD_ECG_1[key], dictionary_HFD_ECG_2[key]]
        dictionary_ages[key] = age_groups[ecg.AgeGroup]










        if (ECG_count_per_age_group_dictionary.keys().__contains__(age_groups[ecg.AgeGroup])):
            ECG_count_per_age_group_dictionary[age_groups[ecg.AgeGroup]] += 1
        else:
            ECG_count_per_age_group_dictionary[age_groups[ecg.AgeGroup]] = 1
    """


    """
    
    print(age_category_ids_dictionary)
    print(HFD_average_by_age_range)
    #print(dictionary_HFD_ECG_1)
    #print(dictionary_HFD_ECG_2)
    #print(dictionary_HFD_ECG_1_2)"""


    # Save age statistics
    """
    
"""


    """
    print("ID: " + row[0])
    print("Higuchi fractal dimension of ECG 1: " + str(HFD_1))

    print("Higuchi fractal dimension of ECG 2: " + str(HFD_2))

    if ((not math.isnan(HFD_1)) and (not math.isnan(HFD_2))):

        if (ECG_per_age_group_dictionary.keys().__contains__(age_groups[row[1]])):
            ECG_per_age_group_dictionary[age_groups[row[1]]] += 1
        else:
            ECG_per_age_group_dictionary[age_groups[row[1]]] = 1

    # HFD_croped = open_record(row[0], 480501)
    # HFD_croped = open_record(row[0], 300000)

    if ((not math.isnan(HFD_1)) and (not math.isnan(HFD_2))):
        dictionary_HFD_ECG_1[row[0]] = HFD_1
        dictionary_HFD_ECG_2[row[0]] = HFD_2
        dictionary_ages[row[0]] = age_groups[row[1]]


line_count += 1

print(f'Processed {line_count} lines.')



with open('number_of_ECG_per_age_range.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for key in ECG_per_age_group_dictionary.keys():
        spamwriter.writerow([key, ECG_per_age_group_dictionary[key]])

    """
def localize_floats(row):
    return str(row).replace('.', ',') if isinstance(row, float) else row

def breaked_ECGs():
    """ECG's with breakes (empties in ECG line)"""

    # Id's of breacked first ecg's
    breaked_first_ecg_ids = []
    # Id's of breacked second ecg's
    breaked_second_ecg_ids = []

    with open('breaked_list/breaked_first_ecg.txt', 'r', encoding='utf-8') as file:
        for str in file:
            breaked_first_ecg_ids.append(str.strip())  # strip - remove spaces and '\n'

    with open('breaked_list/breaked_second_ecg.txt', 'r', encoding='utf-8') as file:
        for str in file:
            breaked_second_ecg_ids.append(str.strip())

    print("Breaked_first_ecg_ids: ", breaked_first_ecg_ids)
    print("Breaked_second_ecg_ids: ", breaked_second_ecg_ids)

    return breaked_first_ecg_ids, breaked_second_ecg_ids

def sets_with_breaked_ECGs(breaked_first_ecg_ids, breaked_second_ecg_ids):
    """General for two ECG's and unique of each ECG"""

    # General for two lists
    general = list(set(breaked_first_ecg_ids) & set(breaked_second_ecg_ids))  # Пересечение множеств
    general.sort()
    print("General: ", general)

    first_unique = list(set(breaked_first_ecg_ids) - set(general))  # First unique from general
    second_unique = list(set(breaked_second_ecg_ids) - set(general))  # Second unique from general
    first_unique.sort()
    second_unique.sort()
    print("First unique: ", first_unique)
    print("Second unique: ", second_unique)

    return general, first_unique, second_unique

#######################################################################################################################
############################################## OPENING RECORDS ########################################################
#######################################################################################################################

def extract_ages (keys, is_remotely=False):

    ages_dictionary = {}

    # Check, if dataset is remotely located
    if is_remotely:
        path = csv_info_file
    else:
        path = path_to_dataset_folder + '/' + csv_info_file

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # Получаем первую строку для инициализации DATABASE_ATTRIBUTES
        first_row = next(csv_reader)

        # Setting counter for first row
        line_count = 0

        # Обрабатываем оставшиеся строки
        for row in csv_reader:
            line_count += 1

            if row[0] in keys:
                ages_dictionary[row[0]] = row[1]

    return ages_dictionary
def read_ECGs_annotation_data(is_remotely, except_breaked):

        """ Open csv info file, print header and information for each record.
        Then fill ECG DATABASE."""

        # Path to CSV file with annotation
        path=""

        breaked_first_ecg_ids, breaked_second_ecg_ids = breaked_ECGs()
        general, first_unique, second_unique = sets_with_breaked_ECGs(breaked_first_ecg_ids, breaked_second_ecg_ids)

        # Check, if dataset is remotely located
        if is_remotely:
            path = csv_info_file
        else:
            path = path_to_dataset_folder + '/' + csv_info_file

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            # Получаем первую строку для инициализации DATABASE_ATTRIBUTES
            first_row = next(csv_reader)

            # Setting counter for first row
            line_count = 0

            for col in first_row:
                DATABASE_ATTRIBUTES.append(col)

            # Обрабатываем оставшиеся строки
            for row in csv_reader:

                line_count += 1

                # Union of general and first_unique (breaked first ECG)
                breaked_first_ecg = list(set(general) | set(first_unique))
                breaked_first_ecg.sort()


                # Check, if ECG in first ecg breaked list
                if (row[0] in breaked_first_ecg):
                    continue

                # 780 - 800
                if (line_count < 818 or line_count > 900):
                    continue

                # If Id is not available
                if (row[0] == 'NaN'):
                    continue

                # If age category is not available. For future maybe do self-organizing without ages.
                if (row[1] == 'NaN'):
                    continue
                else:
                    # Open record returns ecg_1 and ecg_2
                    ecg_s = open_record(row[0], 0, None, remotely=is_remotely)

                    if ecg_s is None:
                        continue

                    # Signal with first ECG
                    signal = ecg_s[0]

                    # Частота дискретизації
                    sampling_rate = 1000

                    r_peaks, rr_intervals = extract_RR_intervals_time_series_and_plot_them(signal, sampling_rate, row[0])

                    #plot_R_peaks(r_peaks, singnal)







                    ###################################################################################################
                    #################################### TO RR - intervals ############################################

                    # Предобработка neurokit2
                    #ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
                    """
                    # Детекция R-зубцов
                    r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

                    #Получаем индексы R - зубцов
                    peaks = _["ECG_R_Peaks"].astype(int)

                    # Проверяем, что индексы находятся в пределах сигнала
                    if np.any(peaks >= len(signal)) or np.any(peaks < 0):
                        raise ValueError("Некоторые индексы R-зубцов выходят за пределы сигнала.")

                    print("Тип peaks:", type(peaks))

                    # Дополнительно: сохранение в файл
                    np.savetxt("rr_intervals/peaks_{0}.txt".format(row[0]), peaks,
                               header="Peaks (s)", comments='', fmt="%.6f")

                    # Преобразуем индексы в временные метки (в секундах)
                    #r_timestamps = np.array(peaks) / sampling_rate
                    print("Peaks in milliseconds: ",peaks)
                    # Вычисляем R-R интервалы (в милисекундах)
                    rr_intervals = np.diff(peaks)

                    # Вывод R-R интервалов
                    print("R-R интервалы (в милисекундах):", rr_intervals)

                    # Дополнительно: сохранение в файл
                    np.savetxt("rr_intervals/rr_intervals_{0}.txt".format(row[0]), rr_intervals, header="RR Intervals (s)", comments='', fmt="%.6f")


                    # Plotting bandpassed signal
                    plt.figure(figsize=(20, 4), dpi=100)
                    plt.xticks(np.arange(0, len(ecg_cleaned) + 1, 150))
                    plt.plot(ecg_cleaned)
                    plt.xlabel('Samples')
                    plt.ylabel('MLIImV')
                    plt.title("Cleaned signal")





                    # Расчет R-R интервалов
                    #rr_intervals = nk.ecg_interval(r_peaks, sampling_rate=sampling_rate)

                    # Сохранение результата
                    #rr_intervals.to_csv("rr_intervals.csv", index=False)


                    # Plotting the R peak locations in ECG signal
                    plt.figure(figsize=(20, 4), dpi=100)
                    plt.xticks(np.arange(0, len(signal) + 1, 150))
                    plt.plot(signal, color='blue')
                    for p in peaks:
                        plt.scatter(p, signal[p], color='red', s=50, marker='*')
                    plt.xlabel('Samples')
                    plt.ylabel('MLIImV')
                    plt.title("R Peak Locations")
                    

                    """

                    # Сохранение результата
                    #rr_intervals.to_csv("rr_intervals.csv", index=False)


                    # Baseline wander of the signal
                    #baseline = bwr.calc_baseline(signal)

                    # Remove baseline from original signal
                    #ecg_out = (signal - baseline)



                    #QRS_detector = pan_tompkins.Pan_Tompkins_QRS()
                    #ecg = pd.DataFrame(np.array([list(range(len(ecg_out))), ecg_out]).T, columns=['TimeStamp', 'ecg'])
                    #output_signal = QRS_detector.solve(ecg)

                    #Низкочастотный фильтр
                    # Параметры фильтра

                    #cutoff = 15.0  # Граничная частота (Гц)
                    #fs = 1000.0  # Частота дискретизации (Гц)
                    #order = 4

                    # Исходный сигнал (например, из ваших данных)
                    # signal = ...

                    # Фильтрация
                    #filtered_signal = butter_lowpass_filter(ecg_out, cutoff, fs, order)

                    #plot_simulationusly_baseline_wander_without_it_and_low_pass_filter(signal, baseline, ecg_out, filtered_signal)





                    #plot_tompkins(pan_tompkins.bpass, pan_tompkins.der, pan_tompkins.sqr, pan_tompkins.mwin)
                    #calculate_heart_rate(ecg)


                    # Calling constructor for RECORD and automatically saving to DATABASE
                    #record = RECORD(row[0], row[1], row[2], row[3], row[4], row[5])
                    #test_record_for_breaks(record)


                #if (not (row[0] == 'NaN')):
                #   selected_records.append(row[0])


def extract_RR_intervals_time_series_and_plot_them(signal, sampling_rate, id):
    """BIO SPPY library for extracting RR-intervals from ECG signal
        input:
            signal - ECG signal
            sampling_rate - sampling_rate
            Id - id of record

    """

    # signal, mdata = storage.load_txt('./examples/ecg.txt')

    # Додання 300 відліків зліва та справа
    extended_signal = np.pad(signal, (300, 300), mode='edge')
    out = ecg.ecg(signal=extended_signal, sampling_rate=sampling_rate, show=False)
    r_peaks = out['rpeaks']  # Отримання індексів R-піків

    # Дополнительно: сохранение в файл
    np.savetxt("rr_peaks/peaks_{0}.txt".format(id), r_peaks,
               header="Peaks (s)", comments='', fmt="%.6f")

    # Process it and plot
    out = ecg.ecg(signal=extended_signal, sampling_rate=sampling_rate, show=True)

    # Вычисляем R-R интервалы (в милисекундах)
    rr_intervals = np.diff(r_peaks)

    np.savetxt("rr_intervals/rr_intervals_{0}.txt".format(id), rr_intervals,
               header="RR Intervals (s)", comments='', fmt="%.6f")

    return r_peaks, rr_intervals

def plot_R_peaks(picks, signal):
    # Plotting the R peak locations in ECG signal
    plt.figure(figsize=(20, 4), dpi=100)
    plt.xticks(np.arange(0, len(signal) + 1, 150))
    plt.plot(signal, color='blue')
    for p in r_peaks:
        plt.scatter(p, signal[p], color='red', s=50, marker='*')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.title("R Peak Locations")


def calculate_heart_rate(ecg):
    # Convert ecg signal to numpy array
    signal = ecg.iloc[:,1].to_numpy()

    # Find the R peak locations
    hr = pan_tompkins.heart_rate(signal,pan_tompkins.fs)
    result = hr.find_r_peaks()
    for r in result:
        print(r)
    result = np.array(result)

    # Clip the x locations less than 0 (Learning Phase)
    result = result[result > 0]

    # Calculate the heart rate
    heartRate = (60*pan_tompkins.fs)/np.average(np.diff(result[1:]))
    print("Heart Rate",heartRate, "BPM")

    # Plotting the R peak locations in ECG signal
    plt.figure(figsize = (20,4), dpi = 100)
    plt.xticks(np.arange(0, len(signal)+1, 150))
    plt.plot(signal, color = 'blue')
    plt.scatter(result, signal[result], color = 'red', s = 50, marker= '*')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.title("R Peak Locations")

    #!!!!!!!!!!!!!!!!!!!!!!!!!
    #res = np.diff(result[1:])
    #for r in res:
    #    print(r)
    print(signal[result])


def plot_tompkins(bpass, der, sqr, mwin):

    # Plotting bandpassed signal
    plt.figure(figsize = (20,4), dpi = 100)
    plt.xticks(np.arange(0, len(bpass)+1, 150))
    plt.plot(bpass[32:len(bpass)-2])
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.title("Bandpassed Signal")

    # Plotting derived signal
    plt.figure(figsize = (20,4), dpi = 100)
    plt.xticks(np.arange(0, len(der)+1, 150))
    plt.plot(der[32:len(der)-2])
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.title("Derivative Signal")

    # Plotting squared signal
    plt.figure(figsize = (20,4), dpi = 100)
    plt.xticks(np.arange(0, len(sqr)+1, 150))
    plt.plot(sqr[32:len(sqr)-2])
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.title("Squared Signal")

    # Plotting moving window integrated signal
    plt.figure(figsize = (20,4), dpi = 100)
    plt.xticks(np.arange(0, len(mwin)+1, 150))
    plt.plot(mwin[100:len(mwin)-2])
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.title("Moving Window Integrated Signal")

from scipy.signal import butter, filtfilt

# Низкочастотный фильтр
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y








def plot_simulationusly_baseline_wander_without_it_and_low_pass_filter(signal, baseline, ecg_out, low_pass_filtered):

    """Plot signal, baseline on first plot. Plot output signal without baseline on the second plot.

    :param signal: input ECG signal
    :param baseline: baseline wander
    :param ecg_out: ECG signal with baseline wander removal
    :return:
    """


    """
    # Generate example signals (replace with your ECG data)
    fs = 500  # Sampling frequency (Hz)
    duration = 10  # Signal duration (seconds)
    t = np.linspace(0, duration, fs * duration)  # Time axis
    signal1 = np.sin(2 * np.pi * 1 * t)  # First signal
    signal2 = np.sin(2 * np.pi * 1 * t + np.pi / 4)  # Second signal (phase shifted)
    """

    # Create the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    # Plot the signals

    ax1.plot(signal, "b-", label="signal")
    ax1.plot(baseline, "r-", label="baseline")
    ax2.plot(ecg_out, "b-", label="signal - baseline")
    ax3.plot(low_pass_filtered, label="Фильтрованный сигнал", linewidth=2)
    # Визуализация

    #plt.plot(time, signal, label="Исходный сигнал", alpha=0.6)






    # Set labels and legends
    ax1.set_title("ECG baseline wander signal")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.legend()
    ax2.set_title("ECG baseline wander removal signal")
    ax2.set_ylabel("Amplitude (mV)")
    ax2.legend()

    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Амплитуда")
    ax3.set_title("Удаление шума низкочастотным фильтром")
    ax3.legend()
    # Enable grid
    ax1.grid()
    ax2.grid()
    ax3.grid()

    # State flag to prevent recursion
    synchronizing = False

    # Synchronize zooming by linking the axes
    def on_xlim_changed(event_ax):
        nonlocal synchronizing
        if synchronizing:
            return  # Prevent recursion

        synchronizing = True  # Set the flag to avoid recursion

        # Synchronize x-limits
        if event_ax == ax1:
            ax2.set_xlim(ax1.get_xlim())
            ax3.set_xlim(ax1.get_xlim())
        elif event_ax == ax2:
            ax1.set_xlim(ax2.get_xlim())
            ax3.set_xlim(ax2.get_xlim())
        elif event_ax == ax3:
            ax1.set_xlim(ax3.get_xlim())
            ax2.set_xlim(ax3.get_xlim())

        synchronizing = False  # Reset the flag after synchronization

    # Connect the zoom synchronization to x-limits changes
    ax1.callbacks.connect("xlim_changed", on_xlim_changed)
    ax2.callbacks.connect("xlim_changed", on_xlim_changed)
    ax3.callbacks.connect("xlim_changed", on_xlim_changed)
    # Show the interactive plot
    plt.show()

def open_record_wfdb(id, min_point, max_point, remotely):
    """Open record with wfdb"""
    record = None

    if remotely:
        record = wfdb.rdrecord(id, min_point, max_point, [0, 1], pn_dir='autonomic-aging-cardiovascular')
    else:
        record = wfdb.rdrecord(
            path_to_dataset_folder + '/' + id, min_point, max_point, [0, 1])

    return record

def open_record(id, min_point, max_point, remotely):

    """ Open each record with ECGs by Id

        Input parameters:
            - Id - id of record
            - min_point - minimum point, at which starts ECG (including this point)
            - max_point - maximum point, at which ends ECG (not including this point)

        Output parameters:
            - [sequence_1, sequence_2] - list with sequence_1 for first ECG and sequence_2 for second ECG

            Describing:
                wfdb.rdrecord(path + '/' + id, min_point, max_point, [0, 1])

                min_point = 0 - The starting sample number to read for all channels
                                (point from what graphic starts (min_point)).

                max_point = None - The sample number at which to stop reading for all
                channels (max_point). Reads the entire duration by default.

                [0, 1] - first two channels (ECG 1, ECG 2); [0] - only first ECG.
            """

    record = None

    if min_point < 0:
        print("Too low minimal point of ECG! Now minimal point is 0!")
        min_point = 0

    if os.path.isfile(path_to_dataset_folder + '/' + id + '.hea') or os.path.isfile(path_to_dataset_folder + '/' + id + '.dat'):
        try:
            record = open_record_wfdb(id, min_point, max_point, remotely)

        except:
            max_point = None
            record = open_record_wfdb(id, min_point, max_point, remotely)
            print("Too hight maximal point of ECG! Now maximal point is None!")
    else:
        print("File with record doesn't exist!")
        return None

    #display(record.__dict__)


    sequence_1 = []
    sequence_2 = []


    # print(record.p_signal)

    for x in record.p_signal:

        # Use first ECG
        sequence_1.append(x[0])

        # Use second ECG
        sequence_2.append(x[1])


    print("Length of first ECG with id {0}: {1}".format(id, str(len(sequence_1))))
    print("Length of second ECG with id {0}: {1}".format(id, str(len(sequence_2))))
    #print(sequence)

    #create_breaks_file(sequence_1, sequence_2)
    wfdb.plot_wfdb(record, title='Record ' + id + ' from Physionet Autonomic ECG')
    #n = input()

    return [sequence_1, sequence_2]



#####################################################################################################################
###################################### TEST RECORD FOR BREAKS #######################################################
#####################################################################################################################

def test_record_for_breaks(sequence_1, sequence_2):
    """Test record for empty points
        input:
            sequence_1: First ECG sequence
            sequence_2: Second ECG sequence
        output:
            breaks1_list: Empty spaces of first ECG
            breaks2_list: Empty spaces of second ECG
    """

    breaks1_list = []
    breaks2_list = []

    for index, point in enumerate(sequence_1):
        if math.isnan(point):
            breaks1_list.append(index)
            #print("Record with first ECG has break at point {0}".format(index))

    for index, point in enumerate(sequence_2):
        if math.isnan(point):
            breaks2_list.append(index)
            #print("Record with second ECG has break at point {0}".format(index))


    if(len(breaks1_list) > 0):
        print("Breaks with first: True")
    if(len(breaks2_list) > 0):
        print("Breaks with second: True")
    return breaks1_list, breaks2_list

"""
def test_records_for_breaks():
    for key in DATABASE.keys:
        record = DATABASE[key]
        for point in record.ECG_1:
            if point == math.nan:
                print("Record {0} has break", key)
        for point in record.ECG_2:
            if point == math.nan:
                print("Record {0} has break", key)
"""
def find_consecutive_breaks(points):
    """
    Identifies consecutive breaks in a list of points.

    :param points: List of break points (sorted integers).
    :return: List of tuples representing consecutive ranges.
    """
    if not points:
        return []

    consecutive_ranges = []
    start = points[0]
    prev = points[0]

    for point in points[1:]:
        if point == prev + 1:
            # Continue the consecutive range
            prev = point
        else:
            # End of a consecutive range
            consecutive_ranges.append((start, prev))
            start = point
            prev = point

    # Append the last range
    consecutive_ranges.append((start, prev))
    return consecutive_ranges


def create_breaks_file(sequence_1, sequence_2):
    """Create breaks file"""
    f = open("breaked_list/breaks_new.txt", "a")

    breaks_1, breaks_2 = test_record_for_breaks(sequence_1, sequence_2)

    consecutive_breaks1 = find_consecutive_breaks(breaks_1)
    consecutive_breaks2 = find_consecutive_breaks(breaks_2)

    print(consecutive_breaks1)
    print(consecutive_breaks2)

    if len(consecutive_breaks1) > 0 or len(consecutive_breaks2) > 0:
        f.write(id)
        f.write("\n")
        f.write("\n")
        f.write("Length of first ECG with id {0}: {1}".format(id, str(len(sequence_1))))
        f.write("\n")
        f.write("Length of second ECG with id {0}: {1}".format(id, str(len(sequence_2))))
        f.write("\n")
        f.write("\n")
        if (len(breaks_1) > 0):
            f.write("Breaks with first: {0}".format(True))
        f.write("\n")
        if (len(breaks_2) > 0):
            f.write("Breaks with second: {0}".format(True))
        f.write("\n")
        f.write("\n")
        if (len(consecutive_breaks1) > 0):
            f.write("Consecutive_breaks 1: ")
        f.write("\n")
        f.write("\n")
        for c_b in consecutive_breaks1:
            f.write(str(c_b))
            f.write("\n")
        f.write("\n")
        f.write("\n")
        if (len(consecutive_breaks2) > 0):
            f.write("Consecutive_breaks 2: ")
        f.write("\n")
        f.write("\n")
        for c_b in consecutive_breaks2:
            f.write(str(c_b))
            f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
    f.close()

    
######################################################################################################################
######################################################################################################################
######################################################################################################################

def save_to_csv(id, sequences, filename):
    """Save ECG sequences to CSV format with time and ecg values"""
    sequence_1, sequence_2 = sequences
    start_time = 0
    sampling_rate = 1000  # Assuming the sampling rate is 1000 Hz, adjust if different

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time", "ecg"])

        for i, value in enumerate(sequence_1):
            time = start_time + i / sampling_rate
            writer.writerow([time, value])

    print(f"ECG data saved to {filename}")

"""

        # parameters: standart_length, cut_method, minutes_passed, count

        with keys without
            passes and with two ECG data. Records without age range are not added in dictionary. ECG data may be with passes, so it must be checked by HFD

            Check if minutes_passed < standart_length

        HFD_OF_ECG_1_AND_2 = {}
        AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES = {}
        ECG_COUNT_PER_EACH_AGE_GROUP = {}
        HIGUCHI_AVERAGE_PER_EACH_AGE_GROUP = {}
        AGE_INDEX_FOR_IDS_OF_BOTH_HFD_VALUES = {}   # ???

        SEXES_FOR_IDS_OF_BOTH_HFD_VALUES = {}
        BMIS_FOR_IDS_OF_BOTH_HFD_VALUES = {}
        LENGTH_FOR_IDS_OF_BOTH_HFD_VALUES = {}

        minutes_points_from_ECG_start = 60000 * minutes_passed


        ############################################## OPTIMIZE !!! ###################################################
        selected_records = []

       

        ###############################################################################################################



        

        record = open_record('1000', 0, None)
            
                    
                    # Check if id and age_group is not NaN

                    if(row[0]>'0100'):
                        continue
                    
                   
                    if (row[0] in selected_records):
                        ###################################print(f'\tId: {row[0]}; Age_group: {age_groups[row[1]]}; Sex: {gender[row[2]]}; BMI: {row[3]}; Length: {row[4]}; Device: {device[row[5]]}.')

                        length = find_length_of_ECGs_in_record(row[0])

                        print("Points in ECG: " + str(length))

                        record = None

                        Check, if ECG length > 1 min. Check this method
                        if ():
                            line_count += 1
                            continue

                        ### Select method of cut of ECG ###

                        ### Warning !!! For ECG length more than 1 min





                        # Возможно вынести minutes-pints_from_ECG-start

                        if (cut_method == TypeOfECGCut.full and length > (minutes_points_from_ECG_start + 5)):

                            record = open_record(row[0], minutes_points_from_ECG_start, None)








                        if (cut_method == TypeOfECGCut.start and ((minutes_points_from_ECG_start + 5)) < standart_length and (length >= standart_length)):
                            record = open_record(row[0], minutes_points_from_ECG_start, standart_length)

                        if (cut_method == TypeOfECGCut.end and ((minutes_points_from_ECG_start + 5)) < standart_length and (length >= standart_length)):

                            delta = 0

                            if ((length - standart_length) >= minutes_points_from_ECG_start):
                                pass
                            else:
                                delta = length - standart_length - minutes_points_from_ECG_start

                            record = open_record(row[0], length - standart_length - delta, None)

                        if (cut_method == TypeOfECGCut.middle and (length >= standart_length)):

                            # Test this case for reliable situation.
                            # If left_length and right length is different cut windows is translated left on 1 point than right
                            left_length = (length - minutes_points_from_ECG_start  - standart_length) // 2

                            if (left_length >= 0):

                                record = open_record(row[0], minutes_points_from_ECG_start + left_length, minutes_points_from_ECG_start + left_length + standart_length)
                            else:
                                record = open_record(row[0], minutes_points_from_ECG_start, standart_length)


                        if not isinstance(record, list):

                            continue

                        ################################################
                        #ecg = RECORD(row[0], row[1], row[2], row[3], row[4], row[5], record[0], record[1])

                        #if(row[0]<'0162'):

                        #old_ecg_dictionary[row[0]] = ecg


                        result = calculate_higuchi(record[0], record[1])


<<<<<<< HEAD
                        ### Plot k and L ##############################################################################

                        k, L = HiguchiFractalDimension.curve_length(np.array(record[0]), opt=True, num_k=50,
                                                                    k_max=None)


                        plt.plot(np.log2(k), np.log2(L), 'ro')
                        plt.ylabel('some numbers')
                        plt.show()

                        ###############################################################################################
=======


>>>>>>> a7e22fc04678f933cea1483c155fdcd1e8416900

                        #For the case, when HFD_1 and HFD_2 simulationusly

                        if result != None:
                            HFD_OF_ECG_1_AND_2[row[0]] = result
                            AGE_INDEX_FOR_IDS_OF_BOTH_HFD_VALUES[row[0]] = row[1]
                            AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES[row[0]] = age_groups[row[1]]
                            SEXES_FOR_IDS_OF_BOTH_HFD_VALUES[row[0]] = gender[row[2]]
                            BMIS_FOR_IDS_OF_BOTH_HFD_VALUES[row[0]] = row[3]
                            LENGTH_FOR_IDS_OF_BOTH_HFD_VALUES[row[0]] = row[4]

                            # Count number of ECG's with both HFD values per each age group

                            if (ECG_COUNT_PER_EACH_AGE_GROUP.keys().__contains__(age_groups[row[1]])):
                                ECG_COUNT_PER_EACH_AGE_GROUP[age_groups[row[1]]] += 1
                            else:
                                ECG_COUNT_PER_EACH_AGE_GROUP[age_groups[row[1]]] = 1

                        if(line_count>10000):
                            break


                    #############################################line_count += 1



        #RECORD.DATABASE[cut_method] = old_ecg_dictionary

        print(HFD_OF_ECG_1_AND_2)
        write_HFD_calculated_values_to_csv(HFD_OF_ECG_1_AND_2, AGE_INDEX_FOR_IDS_OF_BOTH_HFD_VALUES, AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES, cut_method, SEXES_FOR_IDS_OF_BOTH_HFD_VALUES,BMIS_FOR_IDS_OF_BOTH_HFD_VALUES,LENGTH_FOR_IDS_OF_BOTH_HFD_VALUES, minutes_passed)



        ##########################################################################################################
        ##########################################################################################################

        AGE_CATEGORIES_WITH_IDS = {}

        # Fill dictionary with age ranges as keys and lists of id's as values. For each age range list of id's

        for id in AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES.keys():

            if AGE_CATEGORIES_WITH_IDS.keys().__contains__(AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES[id]):
                AGE_CATEGORIES_WITH_IDS[AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES[id]].append(id)
            else:
                AGE_CATEGORIES_WITH_IDS[AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES[id]] = [id]


        HIGUCHI_AVERAGE_PER_EACH_AGE_GROUP = {}



        for age_range in AGE_CATEGORIES_WITH_IDS.keys():

            HFD_1_average = 0
            HFD_2_average = 0

            for id in AGE_CATEGORIES_WITH_IDS[age_range]:
                HFD_1_average += HFD_OF_ECG_1_AND_2[id][0]
                HFD_2_average += HFD_OF_ECG_1_AND_2[id][1]

            length_of_age_categories_with_id_list = len(AGE_CATEGORIES_WITH_IDS[age_range])
            HIGUCHI_AVERAGE_PER_EACH_AGE_GROUP[age_range] = [HFD_1_average / length_of_age_categories_with_id_list, HFD_2_average / length_of_age_categories_with_id_list]


            #################################################################################################################
            #################################################################################################################

        write_number_of_ECGs_per_age_range_for_both_HFD(ECG_COUNT_PER_EACH_AGE_GROUP, cut_method)
        write_average_HFD_values_for_each_age_range(HIGUCHI_AVERAGE_PER_EACH_AGE_GROUP, cut_method)

        print(AGE_CATEGORIES_WITH_IDS)
        print(HIGUCHI_AVERAGE_PER_EACH_AGE_GROUP)
        #print(dictionary_HFD_ECG_1)
        #print(dictionary_HFD_ECG_2)
        #print(dictionary_HFD_ECG_1_2)

        #RECORD.print_database()
"""

def write_HFD_calculated_values_to_csv(hfd_of_ecg_1, age_indexes_for_id, age_ranges_for_id):
    # ECG 1 and 2 simulationusly

    with open('output/HFD_calculated.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in hfd_of_ecg_1.keys():
            spamwriter.writerow([key, age_indexes_for_id[key], age_ranges_for_id[key], localize_floats(hfd_of_ecg_1[key])])

        """, RECORD.DATABASE[type_of_ecg_cut][key].Sex,
                                 RECORD.DATABASE[type_of_ecg_cut][key].BMI]"""



def write_number_of_ECGs_per_age_range_for_both_HFD(ecg_count_per_each_age_group, type_of_ecg_cut):
    with open('number_of_ECGs_per_each_age_range_' + type_of_ecg_cut.name + '_cut.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in ecg_count_per_each_age_group.keys():
            spamwriter.writerow([key, ecg_count_per_each_age_group[key]])
def write_average_HFD_values_for_each_age_range(higuchi_average_per_each_age_group):
    with open('HFD_average_of_ECG_per_age_range.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in higuchi_average_per_each_age_group.keys():
            spamwriter.writerow([key, localize_floats(higuchi_average_per_each_age_group[key])])
"""            
def write_average_HFD_values_for_each_age_range(higuchi_average_per_each_age_group, type_of_ecg_cut):
    with open('HFD_average_of_ECG_per_age_range_' + type_of_ecg_cut.name + '_cut.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in higuchi_average_per_each_age_group.keys():
            spamwriter.writerow([key, localize_floats(higuchi_average_per_each_age_group[key][0]),
                                     localize_floats(higuchi_average_per_each_age_group[key][1])])"""





#def open_record(id, min_point, max_point):

""" Open each record with ECG by Id

        Input parapeters:
            - Id - id of record
            - min_point - minimum point, at which starts ECG (including this point)
            - max_point - maximum point, at which ends ECG (not including this point)"""

# wfdb.rdrecord(... [0, 1] - first two channels (ECG 1, ECG 2); [0] - only first ECG
# 0 - The starting sample number to read for all channels (point from what graphic starts (min_point)).
# None - The sample number at which to stop reading for all channels (max_point). Reads the entire duration by default.
"""
    try:
        record = wfdb.rdrecord(
            path + '/' + id, min_point, max_point, [0, 1])
    except:
        return math.nan





    #display(record.__dict__)







    # print(record.p_signal)

    for x in record.p_signal:

        # Use first ECG
        sequence_1.append(x[0])

        # Use second ECG
        sequence_2.append(x[1])

    print("Initial length of first ECG: " + str(len(sequence_1)))
    #print(sequence)



    ########################## VISUALIZE DISTRIBUTION OF CURVE LENGTH ############################

    k, L = HiguchiFractalDimension.curve_length(sequence_1, opt=True, num_k=50, k_max=None)  # 49 points not 50

    plt.plot(np.log2(k), np.log2(L), 'bo')
    plt.show()

    k, L = HiguchiFractalDimension.curve_length(sequence_2, opt=True, num_k=50,
                                                k_max=None)  # 49 points not 50

    plt.plot(np.log2(k), np.log2(L), 'bo')
    plt.show()

    ###############################################################################################




    wfdb.plot_wfdb(record, title='Record' + id + ' from Physionet Autonomic ECG')


    return [sequence_1, sequence_2]


"""
"""
def convert_record_psignal_to_sequences(p_signal):

    tuple = []
    sequence_1 = []
    sequence_2 = []

    for x in p_signal:

        for i in len(x):

            tuple[]
        # Use first ECG
        sequence_1.append(x[0])

        # Use second ECG
        sequence_2.append(x[1])
"""

def update_id_of_records(old_ecg_dictionary):
    """"""

    ECG_dictionary = {}

    Id = 1

    for key in old_ecg_dictionary.keys():
        new_ecg_key = str("{:04d}".format(Id))                       # For ECG records to 9999
        ECG_dictionary[new_ecg_key] = old_ecg_dictionary[key]
        ECG_dictionary[new_ecg_key].Id = new_ecg_key
        Id += 1

    return ECG_dictionary

def number_of_ECG_by_each_age_group():

    ECG_per_age_group_dictionary = {}

    with open(path+'/'+ csv_info_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # Check if id and age_group is not NaN
                if (not row[1] == 'NaN'):


                    RECORD = open_record(row[0], 0, None)

                    HFD_1 = HiguchiFractalDimension.hfd(np.array(RECORD[0]), opt=True, num_k=50, k_max=None)
                    HFD_2 = HiguchiFractalDimension.hfd(np.array(RECORD[1]), opt=True, num_k=50, k_max=None)

                    print("ID: " + row[0])
                    print("Higuchi fractal dimension of ECG 1: " + str(HFD_1))

                    print("Higuchi fractal dimension of ECG 2: " + str(HFD_2))

                    if ((not math.isnan(HFD_1)) and (not math.isnan(HFD_2))):

                        if(ECG_per_age_group_dictionary.keys().__contains__(age_groups[row[1]])):
                            ECG_per_age_group_dictionary[age_groups[row[1]]] += 1
                        else:
                            ECG_per_age_group_dictionary[age_groups[row[1]]] = 1


                line_count += 1



    with open('number_of_ECG_per_age_range.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in ECG_per_age_group_dictionary.keys():

            spamwriter.writerow([key,ECG_per_age_group_dictionary[key]])





def find_minimum_length_of_records():
    """Find minimum length of ECG record among all dataset"""

    min_length = math.inf
    id = 1

    with open(path + '/' + csv_info_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # Check if id and age_group is not NaN
                if (row[0] != 'NaN' and row[1] != 'NaN'):


                    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    length = find_length_of_ECGs_in_record(row[0])

                    if(length != None):
                        if(length < min_length):
                            min_length = length
                            id = line_count


                line_count += 1


    print(id)

    # result on dataset 480501
    return min_length

def find_maximum_length_of_records():
    """Find maximum length of ECG record among all dataset"""

    max_length = 0

    with open(path + '/' + csv_info_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # Check if id and age_group is not NaN
                if (row[0] != 'NaN' and row[1] != 'NaN'):
                    length = find_length_of_record(row[0])[1]


                    if(length != None):
                        if(length > max_length):
                            max_length = length


                line_count += 1


    return max_length


########################################################################################################################
####################################### CALCULATE ALL LENGTH OF ACCEPTABLE ECGS ########################################
########################################################################################################################

def find_length_of_ECGs_in_record(id):

    """Open each ECG file. If it not opens return None.

    Maybe make local database to not open file every time... """

    try:
        record = wfdb.rdrecord(
            path + '/' + id, 0, None, [0, 1])
    except:
        return None

    #wfdb.plot_wfdb(record, title='Record' + id + ' from Physionet Autonomic ECG')

    import wfdb.plot.plot as pl

    # May be error if not p_signal

    channels = pl._expand_channels(record.p_signal)

    ####################################################################################################################
    ################################ Make another method to check jumps on ECG signal ##################################
    ####################################################################################################################

    if (check_if_ECG_has_omissions(channels[0], channels[1]) == True):
        return None

    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################

    return len(record.p_signal)

def check_if_ECG_has_omissions(ECG_1, ECG_2):

    """Step (omissions) detection
    For the case when two ECG. Maybe create another method for detecting omissions. If at least one ommision found return"""

    for i in range(0, len(ECG_1)):
        if (math.isnan(ECG_1[i]) or math.isnan(ECG_2[i])):
            print("Channel 1: " + str(ECG_1[i]) + ";  Channel 2: " + str(ECG_2[i]) + ";  Index: " + str(i))

            return True

def find_length_of_all_ECGs():

    length_of_all_ECGs = {}

    with open(path + '/' + csv_info_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:

                # Check if age_group is not NaN         // Maybe make additionally for id too

                if ((not (row[0] == 'NaN')) and (not (row[1] == 'NaN'))):

                    length = find_length_of_ECGs_in_record(row[0])

                    if(length != None):
                        length_of_all_ECGs[row[0]] = length
                        print("id:" + str(row[0]) + "  Length: " + str(length))

    with open('length_of_all_ECGs.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['Id', 'Length'])
        for key in length_of_all_ECGs.keys():
            spamwriter.writerow([key, length_of_all_ECGs[key]])

    return length_of_all_ECGs


########################################################################################################################
########################################################################################################################
########################################################################################################################

def find_average_length_of_records():
    """Find average length of ECG records among all dataset"""

    summ = 0
    count = 0

    with open('length_of_all_ECGs.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:

                # Check if age_group is not NaN         // Maybe make additionally for id too

                if (not (row[0] == 'NaN')):

                    summ += int(row[1])
                    count += 1

    average = summ / count

    return average


def find_id_nearest_to_average_record(average, passed_minutes):

        """Check method correctness!
            Have we to add passed_minutes to average value?"""

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        average += passed_minutes

        distance = math.inf
        id = 0

        with open('length_of_all_ECGs.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')

            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if (not (row[0] == 'NaN')):

                        value = int(row[1])
                        new_distance = value - average

                        new_distance_abs = math.fabs(new_distance)

                        #
                        if new_distance_abs < distance and new_distance <= 0:  # Last condition for rounding to lower value of ECG length sample nearest to average + minutes value
                            distance = new_distance_abs
                            id = row[0]

        return id



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def list_rr_intervals_dir():
    """Get files with rr_intervals"""
    import os

    directory = rr_intervals_folder

    # Фильтрация только файлов
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    for file in files:
        print(file)

    return files

def extract_from_file_rr_time_series(files):
    import re

    rr_time_series_dictionary = {}



    for file in files:
        filename = file

        # Используем регулярное выражение для извлечения числового индекса
        match = re.search(r'_(\d+)\.txt', filename)
        if match:
            index = match.group(1)
            rr_time_series_dictionary[index] = None
            #print("Индекс:", index)

            file_path = rr_intervals_folder + "/" + file
            # Чтение файла, начиная со второй строки
            with open(file_path, "r") as file:
                # Пропускаем первую строку
                next(file)

                # Читаем остальные строки
                rr_intervals = [line.strip() for line in file]

            # Вывод значений
            #for rr in rr_intervals:
            #    print(rr)

            rr_intervals = [int(float(x)) for x in rr_intervals]
            #print(rr_intervals)
            rr_time_series_dictionary[index] = rr_intervals

        else:
            print("Индекс не найден")

    #print(rr_time_series_dictionary)

    return rr_time_series_dictionary


def preprocess_rr_intervals(rr_intervals, mode="fixed_count", count=440, duration=300000):
    """
    Предобработка RR-интервалов: выбор фиксированного количества точек или временного интервала.

    :param rr_intervals: массив RR-интервалов (в мс)
    :param mode: "fixed_count" (фиксированное количество) или "fixed_duration" (фиксированная длительность)
    :param count: количество RR-интервалов (например, 500)
    :param duration: длительность анализа в мс (например, 300000 мс = 5 минут)
    :return: обработанный массив RR-интервалов
    """
    rr_intervals = np.array(rr_intervals)  # Преобразуем в массив numpy
    avg_rr = np.mean(rr_intervals)  # Средний RR-интервал
    hr = 60000 / avg_rr  # ЧСС (уд/мин)

    print(f"Средний RR-интервал: {avg_rr:.2f} мс, ЧСС: {hr:.2f} уд/мин")

    if mode == "fixed_count":
        print("Всего: "+str(len(rr_intervals)))
        print(f"Выбрано {count} RR-интервалов")
        return rr_intervals[:count]  # Берем первые count точек

    elif mode == "fixed_duration":
        total_time = np.cumsum(rr_intervals)  # Суммируем RR-интервалы
        valid_indices = np.where(total_time <= duration)[0]  # Ищем точки, укладывающиеся в duration
        print(f"Выбрано {len(valid_indices)} RR-интервалов (на {duration / 1000} секунд)")
        return rr_intervals[valid_indices]  # Возвращаем только эти точки

    else:
        raise ValueError("Неверный режим. Используйте 'fixed_count' или 'fixed_duration'.")


def check_for_minimum_time_rr_time_intervals(rr_time_series_dictionary, min_time=300000):
    """Перевірити, чи сума інтервалів часового ряду менша 5 хв"""


    for key in rr_time_series_dictionary.keys():
        summ = np.sum(rr_time_series_dictionary[key])
        if summ < min_time:
            print("Запись временных рядов меньше 5 минут!")


if __name__ == '__main__':



    #myarray = np.fromfile("D:/Projects/ECGHiguchi/mit-bih-arrhythmia-database-1.0.0/101.dat", dtype=float)

    #for i in range (0, len(myarray)):
    #    print(myarray[i])

    #record = wfdb.rdrecord('D:/Projects/ECGHiguchi/mit-bih-arrhythmia-database-1.0.0/102', 4)
    #wfdb.plot_wfdb(record, title='Record a01 from Physionet Apnea ECG')
    #display(record.__dict__)



    x = np.random.randn(10000)
    #y = np.empty(9900)
    #for i in range(x.size - 100):
    #    y[i] = np.sum(x[:(i + 100)])

    ## Note x is a Guassian noise, y is the original Brownian data used in Higuchi, Physica D, 1988.

    #print(HiguchiFractalDimension.hfd(x, opt=False))
    # ~ 2.00
    #hfd.hfd(y)  # ~ 1.50


    print_hi('Higuchi!')




    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    #print(find_minimum_length_of_records())                # Result: minimum length of ECG on dataset 480501 points
    #print(find_maximum_length_of_records())                 # Result: maximum length of ECG on dataset 2168200 points

    #open_record('0092',0, None)








    should_additionally_cat_minutes_points = total_minutes_points_from_ECG_start - expected_minutes_points_that_ECG_waited

    ################################## Find length of all ECGs and save to CSV file ###################################

    #find_length_of_all_ECGs()

    ######################################## Find average length of records ###########################################

    #average = find_average_length_of_records()
    #print(average)

    ######################### Знайти індекс запису ЕКГ, по довжині найближчої до середньої #############################

    #id = find_id_nearest_to_average_record(average, total_minutes_points_from_ECG_start - expected_minutes_points_that_ECG_waited)
    #print(id)

    ################################### Довжина ЕКГ в записі з заданим індексом ########################################

    #print(find_length_of_ECGs_in_record(id))  # Result 1057346

    ###################################################################################################################

    #read_ECG_annotation_data(1057346, TypeOfECGCut.full, should_additionally_cat_minutes_points, 5)





    ############################################## !!!!!!!!!!!! ######################################################

    #read_ECGs_annotation_data(False, True)


    num_k_value = 50
    k_max_value = None


    files = list_rr_intervals_dir()
    rr_time_series_dictionary = extract_from_file_rr_time_series(files)

    # 440 min count, all > 5 min
    preprocessed_dictionary = {}

    for key in rr_time_series_dictionary.keys():
        preprocessed_dictionary[key] = preprocess_rr_intervals(rr_time_series_dictionary[key])

    """
    # Find minimum len of rr_time_series
    min_len = 1000000
    for key in rr_time_series_dictionary.keys():
        ln = len(rr_time_series_dictionary[key])
        if ln < min_len:
            min_len = ln
    print(min_len)


    """
    #check_for_minimum_time_rr_time_intervals(rr_time_series_dictionary)




    # Пример RR-интервалов (синтетические данные, реальные данные можно загрузить из файла)
    #np.random.seed(42)
    #rr_intervals = np.random.normal(800, 50, 1000)  # Генерируем 1000 RR-интервалов со средним 800 мс
    """
    for key in rr_time_series_dictionary.keys():
        rr_intervals = rr_time_series_dictionary[key]
        
        # Выбираем фиксированное количество (500 RR-интервалов)
        selected_rr_1 = preprocess_rr_intervals(rr_intervals, mode="fixed_count", count=440)

        # Выбираем фиксированную длительность (5 минут)
        selected_rr_2 = preprocess_rr_intervals(rr_intervals, mode="fixed_duration", duration=300000)

        # Визуализация
        plt.figure(figsize=(12, 5))
        plt.plot(selected_rr_1, label="440 RR-интервалов", alpha=0.7)
        plt.plot(selected_rr_2, label="5 минут записи", linestyle="dashed", alpha=0.7)
        plt.xlabel("Индекс")
        plt.ylabel("RR-интервал (мс)")
        plt.legend()
        plt.title("Выбор RR-интервалов разными методами")
        plt.show()
    """

    calculate_higuchi(preprocessed_dictionary)

    """
    #test_records_for_breaks()
    # Example usage
    path = "/path/to/data"  # Update this path to the location of your data files
    record_id = "0001"
    min_point = 0
    max_point = None

    sequences = [DATABASE['0001'].ECG_1, DATABASE['0001'].ECG_2]
    if sequences is not math.nan:
        save_to_csv(record_id, sequences, f"ecg_{record_id}.csv")
    """


    print_database_attributes()
    print_database()
    
    #print[((RECORD)DATABASE[0]).__str__()]


    #print(find_length_of_record_ECG('0400'))
    #minimum_length = find_minimum_length_of_records()
    #minimum_length = 480501







    # Get dictionary with length of every ECG for each Id


    #length_of_every_ECG = find_length_of_every_ECG()


    #


    #id = find_id_nearest_to_average_record(length_of_every_ECG, average, 3)
    #print(id)  # Result 0067
    #length = find_length_of_record('0067')  # 1057346 value   17,622433 мин
    #minimum_length = length
    #print(minimum_length)

    #ln = find_length_of_record('0006')# '920744'
    #print(ln)
    #read_ECG_data(minimum_length, TypeOfECGCut.full, 1)
    """record = open_record('0006', 920743, None)

    ln = find_length_of_record('0125')# '920744'
    print(ln)
    record = open_record('0125', 902033, None)

    if (record != 'Nan'):
        print(record)

    hfd = calculate_higuchi(record[0],record[1])
    print(hfd)
"""
    #read_ECG_data(1057346, TypeOfECGCut.full, 3)
    #open_record('0637', 0, 480501)
    #number_of_ECG_by_each_age_group()
    #average = find_average_length_of_records(3)
    #read_ECG_data(minimum_length, TypeOfECGCut.full, 3)
    #h = calculate_higuchi([1, 2],[2, 3])
    #print(h)



    ################################################################################################################





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
