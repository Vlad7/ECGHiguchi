"""import wfdb
from wfdb import processing

path_to_dataset_folder = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'

sig, fields = wfdb.rdsamp(r'{0}/0001'.format(path_to_dataset_folder), channels=[0])
xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
xqrs.detect()
wfdb.plot_items(signal=sig, ann_samp=[xqrs.qrs_inds])
"""
import wfdb
import csv
import os
import numpy as np
from biosppy.signals import ecg
import matplotlib.pyplot as plt
from statistics import mean

from neurokit2 import rsp_amplitude
from wfdb import processing
import neurokit2 as nk
import pandas as pd

path_to_dataset_folder = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'
#path_to_dataset_folder  = 'C:/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'

csv_info_file = 'subject-info.csv'

rr_intervals_folder="rr_intervals/all"

DATABASE_ATTRIBUTES = []

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

def open_record_wfdb(id, min_point, max_point, remotely):
    """Open record with wfdb"""
    record = None

    if remotely:
        record = wfdb.rdrecord(id, min_point, max_point, [0, 1], pn_dir='autonomic-aging-cardiovascular')
    else:
        record = wfdb.rdrecord(
            path_to_dataset_folder + '/' + id, min_point, max_point, [0, 1])

    return record

#######################################################################################################################
#################################### EXTRACTING RR INTERVALS ##########################################################
#######################################################################################################################
def extract_RR_intervals_time_series_and_plot_them(signal, sampling_rate, id):
    """BIO SPPY library for extracting RR-intervals from ECG signal
        input:
            signal - ECG signal
            sampling_rate - sampling_rate
            Id - id of record

        output:

            r_peaks, rr_intervals - R peaks and RR intervals

    """

    # signal, mdata = storage.load_txt('./examples/ecg.txt')

    # Додання 500 відліків зліва та справа для коректного розпізнання сигналу 
    extended_signal = np.pad(signal, (500, 500), mode='edge')

    # Для виявлення R-піків використовується бібліотека biosppy
    out = ecg.ecg(signal=extended_signal, sampling_rate=sampling_rate, show=True)
    r_peaks = out['rpeaks']  # Отримання індексів R-піків

    #Відфільтрований сигнал, відступаємо від початку 500 і від кінця 500 (обернена операція до np.pad)
    cleaned_signal = out[1][500:-500]

    # Відступаємо назад на 500 (обернена операція до np.pad)
    r_peaks = r_peaks - 500
    #print(r_peaks)
    # Дополнительно: сохранение в файл
    #np.savetxt("rr_peaks/peaks_{0}.txt".format(id), r_peaks,
    #           header="Peaks (s)", comments='', fmt="%.6f")

    # Process it and plot
    #out = ecg.ecg(signal=extended_signal, sampling_rate=sampling_rate, show=True)

    # Вычисляем R-R интервалы (в милисекундах)
    #rr_intervals = np.diff(r_peaks)

    #np.savetxt("rr_intervals/rr_intervals_{0}.txt".format(id), rr_intervals,
    #           header="RR Intervals (s)", comments='', fmt="%.6f")

    return cleaned_signal, r_peaks



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


    return [sequence_1, sequence_2]

def read_ECGs_annotation_data(is_remotely, except_breaked):
    """ Open csv info file, print header and information for each record.

        input: is_remotely - download annotation file and record remotely from internet

    """
    files = list_files_with_rr_intervals()
    # Ecg's with indexes that have variability
    ids_with_variability = extract_from_files_ids(files)
    print(ids_with_variability)
    # Path to CSV file with annotation
    path = ""

    # Id's of breaked first and second ecg's
    breaked_first_ecg_ids, breaked_second_ecg_ids = breaked_ECGs()

    # Id's general both for first ecg, first unique, second unique
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

        # Union of general and first_unique (breaked first ECG)
        breaked_first_ecg = list(set(general) | set(first_unique))
        breaked_first_ecg.sort()

        # Обрабатываем оставшиеся строки
        for row in csv_reader:

            line_count += 1

            # Check, if ECG in first ecg breaked list
            if (row[0] in breaked_first_ecg):
                continue

            if (row[0] not in ids_with_variability):
                continue
            # 780 - 800; 1081 <
            #if (line_count != 1 and line_count != 2):
            #   continue

            print ("Hello")
            # If Id is not available
            if (row[0] == 'NaN'):
                continue

            # If age category is not available. For future maybe do self-organizing without ages.
            if (row[1] == 'NaN'):
                continue

            else:
                # Open record returns ecg_1 and ecg_2
                ecg_s = open_record(row[0], 0, 480501, remotely=is_remotely)

                if ecg_s is None:
                    continue

                # Signal with first ECG
                ecg_signal = np.array(ecg_s[0])

                # Частота дискретизації
                sampling_rate = 1000

                # Filter signal to cleaned and detect r_peaks
                cleaned_signal, r_peaks = extract_RR_intervals_time_series_and_plot_them(ecg_signal,
                                                sampling_rate, row[0])

                # Next, use NeuroKit for P, Q, S, T (around R)
                # Delineate the ECG signal using neurokit2, cwt with hight precision, for quicker use dwt
                _, waves_peaks = nk.ecg_delineate(cleaned_signal, r_peaks,
                                                 sampling_rate=sampling_rate, method="cwt", show=True)

                calculate_ECG_features(cleaned_signal, r_peaks, waves_peaks)


                # Припустимо, ми аналізуємо перші три серцевих цикли на графіку:

                count_plot = 3
                plot_ECG_parameters(cleaned_signal, waves_peaks, r_peaks, count_plot)










                """
                import matplotlib.pyplot as plt
                import numpy as np


                plt.figure(figsize=(12, 4))
                plt.plot(time, ecg_signal, label="ECG")
                plt.scatter(time[r_peaks], ecg_signal[r_peaks], color='red', label="R-peaks")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.title("ECG with R-peaks")
                plt.legend()
                plt.grid(True)
                plt.show()


                # === Отобразим
                nk.ecg_plot(signals)
                plt.show()"""
                """
                import neurokit2 as nk
                import matplotlib.pyplot as plt

                # 1. Генеруємо синтетичний ЕКГ-сигнал
                #signal = nk.ecg_simulate(duration=10, sampling_rate=1000)

                # 2. Аналіз сигналу: виявлення компонентів (P, QRS, T)
                ecg_signals, info = nk.ecg_process(signal, sampling_rate=1000)

                # 3. Отримання індексів початку і кінця компонентів
                r_peaks = info["ECG_R_Peaks"]
                p_peaks = info["ECG_P_Peaks"]
                t_peaks = info["ECG_T_Peaks"]

                # 4. Візуалізація
                plt.figure(figsize=(15, 5))
                plt.plot(ecg_signals["ECG_Clean"], label="ECG Signal")

                # Позначимо точки на графіку
                plt.scatter(p_peaks, ecg_signals["ECG_Clean"][p_peaks], color="green", label="P peaks")
                plt.scatter(r_peaks, ecg_signals["ECG_Clean"][r_peaks], color="red", label="R peaks")
                plt.scatter(t_peaks, ecg_signals["ECG_Clean"][t_peaks], color="purple", label="T peaks")

                plt.title("ЕКГ-сигнал з позначеними P, R, T")
                plt.xlabel("Час (мс)")
                plt.ylabel("Амплітуда")
                plt.legend()
                plt.grid()
                plt.show()
                """
                # Частота дискретизації
                #sampling_rate = 1000

                #r_peaks, rr_intervals = extract_RR_intervals_time_series_and_plot_them(signal, sampling_rate, row[0])

                """
                import csv

                ecg_attributes = [
                    {"time": 0.0, "amplitude": 0.1, "heart_rate": 75},
                    {"time": 0.01, "amplitude": 0.12, "heart_rate": 75},
                    {"time": 0.02, "amplitude": 0.14, "heart_rate": 76},
                    # и так далее...
                ]

                with open("ecg_data.csv", mode="w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["time", "amplitude", "heart_rate"])
                    writer.writeheader()
                    writer.writerows(ecg_attributes)
                """
def calculate_HCF(r_peaks):
    # Вычисляем временные интервалы между пиками R

    intervals = np.diff(r_peaks) # Находим разности между последовательными пиками

    # Рассчитываем средний интервал (в секундах)
    average_interval = np.mean(intervals)

    # Рассчитываем частоту сердечных сокращений (в ударах в минуту)
    if average_interval > 0:
        heart_rate = 60 / average_interval
    else:
        heart_rate = 0

    print(f"Частота сердечных сокращений: {heart_rate:.2f} уд/мин")

    return heart_rate

def mean_RR_interval(r_peaks):
    intervals = np.diff(r_peaks)  # Находим разности между последовательными пиками

    # Рассчитываем средний интервал (в секундах)
    average_interval = np.mean(intervals)

    return average_interval

def mean_ST_interval(s_waves, t_waves):
    st = t_waves - s_waves
    average_st = np.mean(st)

    return average_st

def mean_QRS_complex(q_waves, s_waves):
    qrs = s_waves - q_waves
    average_qrs = np.mean(qrs)

    return average_qrs

def calculate_ECG_features(cleaned_signal, r_peaks, waves_peaks):

    p_start_waves = pd.Series(waves_peaks["ECG_P_Onsets"])
    p_peaks = pd.Series(waves_peaks["ECG_P_Peaks"])
    p_end_waves   = pd.Series(waves_peaks["ECG_P_Offsets"])
    q_waves = pd.Series(waves_peaks["ECG_Q_Peaks"])
    s_waves = pd.Series(waves_peaks["ECG_S_Peaks"])
    t_start_waves = pd.Series(waves_peaks["ECG_T_Onsets"])
    t_peaks = pd.Series(waves_peaks["ECG_T_Peaks"])
    t_end_waves =   pd.Series(waves_peaks["ECG_T_Offsets"])


    p_start_waves = p_start_waves[p_start_waves.first_valid_index():p_start_waves.last_valid_index() + 1]
    p_peaks = p_peaks[p_peaks.first_valid_index():p_peaks.last_valid_index() + 1]
    p_end_waves = p_end_waves[p_end_waves.first_valid_index():p_end_waves.last_valid_index() + 1]
    q_waves = q_waves[q_waves.first_valid_index():q_waves.last_valid_index() + 1]
    s_waves = s_waves[s_waves.first_valid_index():s_waves.last_valid_index() + 1]
    t_start_waves = t_start_waves[t_start_waves.first_valid_index():t_start_waves.last_valid_index() + 1]
    t_peaks = t_peaks[t_peaks.first_valid_index():t_peaks.last_valid_index() + 1]
    t_end_waves = t_end_waves[t_end_waves.first_valid_index():t_end_waves.last_valid_index() + 1]

    r_peaks = r_peaks[r_peaks.first_valid_index():r_peaks.last_valid_index() + 1]

    ###############################################################################
    # Нужно найти для каждого P соответствующий R позже него, и тогда всё будет ок.

    index_from = 0
    minimal_p_start_wave = p_start_waves[0]

    for peak in r_peaks:
        if peak > minimal_p_start_wave:
            break
        index_from +=1

    q_waves = q_waves[index_from:]
    r_peaks = r_peaks[index_from:]
    s_waves = s_waves[index_from:]

    min_length = min(len(r_peaks), len(p_start_waves))

    # Вирівняти обидва масиви по довжині
    p_start_waves = p_start_waves[:min_length] # Зріз до індексу min_length, не включаючи його
    p_peaks = p_peaks[:min_length]
    p_end_waves = p_end_waves[:min_length]
    r_peaks = r_peaks[:min_length]
    q_waves = q_waves[:min_length]
    s_waves = s_waves[:min_length]



    ###############################################################################
    # Нужно найти для каждого R соответствующий T позже него, и тогда всё будет ок.

    minimal_r_start_wave = r_peaks[0]

    index_from = 0

    for t_wave in t_start_waves:
        if t_wave > minimal_r_start_wave:
            break
        index_from += 1

    t_start_waves = t_start_waves[index_from:]
    t_peaks = t_peaks[index_from]
    t_end_waves  = t_end_waves[index_from:]

    min_length = min(len(r_peaks), len(t_start_waves))

    # Вирівняти обидва масиви по довжині
    r_peaks = r_peaks[:min_length]
    q_waves = q_waves[:min_length]
    s_waves = s_waves[:min_length]
    t_start_waves = t_start_waves[:min_length]  # Зріз до індексу min_length, не включаючи його
    t_peaks = t_peaks[:min_length]
    t_end_waves = t_end_waves[:min_length]
    p_start_waves = p_start_waves[:min_length]
    p_peaks = p_peaks[:min_length]
    p_end_waves = p_end_waves[:min_length]
    ################################################################################
    # Можливо перевірити випадок, коли останні значення не співпадають

    print("P start waves: ", p_start_waves)
    print("P end waves: ", p_end_waves)
    print("Q waves: ", q_waves)
    print("R peaks: ", r_peaks)
    print("S waves: ", s_waves)
    print("T start waves: ", t_start_waves)
    print("T end waves: ", t_end_waves)

    ################### Mask NaN values #######################
    mask = ~np.isnan(p_start_waves)

    p_end_waves = p_end_waves[mask]
    q_waves = q_waves[mask]
    r_peaks = r_peaks[mask]
    s_waves = s_waves[mask]
    t_start_waves = t_start_waves[mask]
    t_end_waves = t_end_wavesp[mask]
    ###########################################################

    p_duration = find_P_interval(p_start_waves, p_end_waves)    #!!!
    t_duration = find_T_interval(t_start_waves, t_end_waves)    #!!!

    ###############################################################################
    # Calculate heart contraction frequency - 1-st parameter (heart rate)
    HCF = calculate_HCF(r_peaks)                        #!!!
    print("Heart rate: ",HCF)

    ################################ Перевіряємо, чи PR інтервали однакові ###########################
    corrected_pr_intervals = corrected_PR_intervals(r_peaks, waves_peaks["ECG_P_Onsets"][1:len(r_peaks) - 1])

    # Перевіримо варіацію:
    std_dev = np.std(corrected_pr_intervals)
    mean_val = np.mean(corrected_pr_intervals)
    coefficient_of_variation = std_dev / mean_val       #!!!

    print(f"PR intervals: {corrected_pr_intervals}")

    # 2-nd and 3-rd parameters (mean PR intervals, CoefVar)
    print(f"Mean PR: {mean_val:.2f}, Std: {std_dev:.2f}, CoefVar: {coefficient_of_variation:.4f}")
    ##################################################################################################

    # 4-th parameter (mean RR intervals)
    mean_RR = mean_RR_interval(r_peaks)

    print(f"Mean RR: ", mean_RR)

    # 5-th parameter (mean ST interval)
    mean_ST = mean_ST_interval(s_waves, t_start_waves)

    print(f"Mean ST: ", mean_ST)

    # 6-th parameter (QRS complex)
    mean_QRS = mean_QRS_complex(q_waves, s_waves)

    # p_end = delineate_info["ECG_P_Offsets"][index]
    # q_start = delineate_info["ECG_Q_Peaks"][index]
    # s_end = delineate_info["ECG_S_Peaks"][index]
    # t_start = delineate_info["ECG_T_Onsets"][index]


    # 7-th and 8-th parameters (P duration, T duration)
    print("P duration: ", p_duration)
    print("T duration: ", t_duration)

    #9-th, 10-th and 11-th parameter
    p_amplitude = np.mean(cleaned_signal[p_peaks])
    r_amplitude = np.mean(cleaned_signal[r_peaks])
    t_amplitude = np.mean(cleaned_signal[t_peaks])

    # print("P end ",p_end) #Index of P end
    # print(q_start)  #Index of q start
    # PQ сегмент:
    # pq_segment = cleaned_signal[p_end:q_start]

    # ST сегмент:
    # st_segment = cleaned_signal[s_end:t_start]
    # print(pq_segment)
    # print(st_segment)

    """
    print(waves_peaks['ECG_T_Peaks'])
    t_peaks = np.array(waves_peaks['ECG_T_Peaks'])
    t_peaks = t_peaks[~np.isnan(t_peaks)].astype(int)

    time = np.linspace(0, len(ecg_signal) / fs, len(ecg_signal))
    """

    # Обчислюємо PR-інтервали
    #pr_intervals = np.array(r_peaks[1:len(r_peaks) - 1]) - waves_peaks["ECG_P_Onsets"][1:len(r_peaks) - 1]

    #for i in range(len(p_start_waves)):
     #   print(f"P: {p_start_waves[i]}, R: {r_peaks[i]}, PR: {r_peaks[i] - p_start_waves[i]}")




    print(r_peaks[1:len(r_peaks) - 1])
    print(p_start_waves)
    # Видаляємо NaN
    #pr_intervals = pr_intervals[~np.isnan(pr_intervals)]


def find_P_interval(p_start_waves, p_end_waves):
    # Отримання початкової та кінцевої точок P-інтервалу для всіх серцевих циклів,
    # окрім першого та останнього

    p_start_waves = np.array(p_start_waves)
    p_end_waves = np.array(p_end_waves)

    p_diff_list = p_end_waves - p_start_waves

    # Видаляємо NaN
    p_diff_list = p_diff_list[~np.isnan(p_diff_list)]
    # Середня тривалість P-інтервалу
    p_duration = mean(p_diff_list)

    return p_duration

def find_T_interval(t_start_waves, t_end_waves):
    # Отримання початкової та кінцевої точок T-інтервалу для всіх серцевих циклів,
    # окрім першого та останнього

    t_start_waves = np.array(t_start_waves)
    t_end_waves = np.array(t_end_waves)

    t_diff_list = t_end_waves - t_start_waves

    # Видаляємо NaN
    t_diff_list = t_diff_list[~np.isnan(t_diff_list)]
    # Середня тривалість T-інтервалу
    t_duration = mean(t_diff_list)

    return t_duration

def plot_ECG_parameters(cleaned_signal, waves_peaks, r_peaks, count_plot):
    # Входные данные (замени своими переменными)
    signal = cleaned_signal[:4000]
    x = np.arange(len(signal))

    # Отрисовка сигнала
    plt.figure(figsize=(12, 6))
    plt.plot(x, signal, label="ECG", color="black")
    plt.scatter(x[waves_peaks['ECG_T_Peaks'][:count_plot]],
                signal[waves_peaks['ECG_T_Peaks'][:count_plot]], color='red', label="R-peaks")

    # Изолиния
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1, label="Изолиния")





    # Custom events (insert your lists)
    def mark_events(event_indices, color, label, style='--'):
        for i in event_indices:
            plt.axvline(x=i, color=color, linestyle=style, linewidth=1.5)
        # Add only once in legend
        if len(event_indices) > 0:
            plt.axvline(x=event_indices[0], color=color, linestyle=style, label=label, linewidth=1.5)

    # Пример: замените списки на ваши
    mark_events(waves_peaks["ECG_P_Onsets"][:count_plot], "green", "P начало")
    mark_events(waves_peaks['ECG_P_Peaks'][:count_plot], "lime", "P пик", style='-.')
    mark_events(waves_peaks["ECG_P_Offsets"][:count_plot], "green", "P конец", style=':')

    mark_events(waves_peaks['ECG_Q_Peaks'][:count_plot], "blue", "Q", style='--')
    mark_events(r_peaks[:count_plot], "red", "R peak", style='--')
    mark_events(waves_peaks['ECG_S_Peaks'][:count_plot], "blue", "S", style='--')

    mark_events(waves_peaks["ECG_T_Onsets"][:count_plot], "purple", "T начало")
    mark_events(waves_peaks['ECG_T_Peaks'][:count_plot], "magenta", "T пик", style='-.')
    mark_events(waves_peaks["ECG_T_Offsets"][:count_plot], "purple", "T конец", style=':')

    # Легенда и стили
    plt.legend(loc='upper right')
    plt.title("Кастомная визуализация зубцов ЭКГ")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def corrected_PR_intervals(r_peaks, p_start_waves):
    # Нужно найти для каждого P соответствующий R позже него, и тогда всё будет ок.

    corrected_pr_intervals = []

    for p in p_start_waves:
        r_candidates = r_peaks[r_peaks > p]
        if len(r_candidates) > 0:
            nearest_r = r_candidates[0]
            corrected_pr_intervals.append(nearest_r - p)

    corrected_pr_intervals = np.array(corrected_pr_intervals)
    return corrected_pr_intervals


def extract_from_files_ids(files):
    """Extract from files RR intervals time series

        input: files - file names
        output: rr_time_series_dictionary - dictionary with id as key and list of RR-intervals as value"""

    import re

    list_of_ids = []

    for file in files:
        filename = file

        # Используем регулярное выражение для извлечения числового индекса
        match = re.search(r'_(\d+)\.txt', filename)
        if match:
            index = match.group(1)
            list_of_ids.append(index)

    return list_of_ids

def list_files_with_rr_intervals():
    """Get list of files with rr_intervals time series from rr_interval/all folder"""
    import os

    directory = rr_intervals_folder

    # Фильтрация только файлов
    files = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    for file in files:
        print(file)

    return files

def write_ECG_parameters_to_csv(sex, hfd_of_ecg_1, age_indexes_for_id, age_ranges_for_id):
    # ECG 1 and 2 simulationusly

    with open('output/{0}_HFD_calculated.csv'.format(sex), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in age_indexes_for_id.keys():
            spamwriter.writerow([key, age_indexes_for_id[key], age_ranges_for_id[key], localize_floats(hfd_of_ecg_1[key])])

read_ECGs_annotation_data(False, True)


