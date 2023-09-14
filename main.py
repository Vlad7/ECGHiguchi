# This is an ECG Higuchi script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Age groups
import math
import pandas as pd
import openpyxl
import enum

gender = {'0': 'male', '1': 'female'}
device = {'0': 'TFM, CNSystems', '1': 'CNAP 500, CNSystems; MP150, BIOPAC Systems'}

age_groups = {'1': '18 - 19',
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


from IPython.display import display
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv

#######################################################################################################################

# Path to dataset of ECG
load_data_online = False
path = 'D:\\SCIENCE\\autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0\\autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'

csv_info_file = 'subject-info.csv'

#######################################################################################################################
minimum_length_of_ECG = 480501
average_length_of_ECG = 1140000


# ECG_dictionary with information about ECG


class RECORD:

    """ Record class that represents necessary information about ECG database """

    # Database with filtered ages

    DATABASE = {}

    def __init__(self, id, age_group, sex, bmi, length, device, ecg_1, ecg_2):

        """Without third time series - pressure"""

        self.Id = id
        self.AgeGroup = age_group
        self.Sex = sex
        self.BMI = bmi
        self.Length = length
        self.Device = device
        self.ECG_1 = ecg_1
        self.ECG_2 = ecg_2

    def __str__(self):

        return str(f'\tId: {self.Id}; Age_group: {self.AgeGroup}; Sex: {self.Sex}; BMI: {self.BMI}; Length: {self.Length}; Device: {self.Device}.')

    def print_database():
        for en in RECORD.DATABASE.keys():
            for x in RECORD.DATABASE[en]:
                print(RECORD.DATABASE[en][x])
                print(RECORD.DATABASE[en][x].ECG_1)
                print(RECORD.DATABASE[en][x].ECG_2)




def calculate_higuchi(ECG_1, ECG_2, num_k_value=50, k_max_value=None):

    """For the case when two ECG.
        Input parameters:
        num_k_value - number of k values
        k_max_value - value of Kmax"""

    """
    dictionary_HFD_ECG_1 = {}
    dictionary_HFD_ECG_2 = {}
    
    dictionary_ages = {}
    """




    ECG_count_per_age_group_dictionary = {}

    Higuchi_average_per_age_group_dictionary = {}



    HFD_1 = HiguchiFractalDimension.hfd(np.array(ECG_1), opt=True, num_k=num_k_value, k_max=k_max_value)
    HFD_2 = HiguchiFractalDimension.hfd(np.array(ECG_2), opt=True, num_k=num_k_value, k_max=k_max_value)

    """
    if (not math.isnan(HFD_1)):
        dictionary_HFD_ECG_1[ecg.Id] = HFD_1

    if (not math.isnan(HFD_2)):
        dictionary_HFD_ECG_2[ecg.Id] = HFD_2"""

    if ((not math.isnan(HFD_1)) and (not math.isnan(HFD_2))):
        return [HFD_1, HFD_2]
    else:
        return None


    # For testing
    #dictionary_HFD_ECG_1.pop("0001")
    #dictionary_HFD_ECG_2.pop("0010")

    # Intersect of two sets
    keys = list(set(dictionary_HFD_ECG_1.keys()) & set(dictionary_HFD_ECG_2.keys()))

    """dictionary_ages = {}
    for key in keys:
        dictionary_HFD_ECG_1_2[key] = [dictionary_HFD_ECG_1[key], dictionary_HFD_ECG_2[key]]
        dictionary_ages[key] = age_groups[ecg.AgeGroup]










        if (ECG_count_per_age_group_dictionary.keys().__contains__(age_groups[ecg.AgeGroup])):
            ECG_count_per_age_group_dictionary[age_groups[ecg.AgeGroup]] += 1
        else:
            ECG_count_per_age_group_dictionary[age_groups[ecg.AgeGroup]] = 1

    age_category_ids_dictionary = {}

    #For each age range list of id's

    for key in dictionary_ages.keys():

        if age_category_ids_dictionary.keys().__contains__(dictionary_ages[key]):
            age_category_ids_dictionary[dictionary_ages[key]].append(key)
        else:
            age_category_ids_dictionary[dictionary_ages[key]] = [key]

    HFD_average_by_age_range = {}

    for key in age_category_ids_dictionary.keys():

        HFD_1_average = 0
        HFD_2_average = 0

        for age_range_key in age_category_ids_dictionary[key]:
            HFD_1_average += dictionary_HFD_ECG_1_2[age_range_key][0]
            HFD_2_average += dictionary_HFD_ECG_1_2[age_range_key][1]

        length_of_age_range_id_list = len(age_category_ids_dictionary[key])
        HFD_average_by_age_range[key] = [HFD_1_average / length_of_age_range_id_list, HFD_2_average / length_of_age_range_id_list]

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

def read_ECG_data(standart_length, cut_method, one_minute_pass):

        """ Open csv info file, print header and information for each record. Then fill ECG_dictionary with keys without
         passes and with two ECG data. Records without age range are not added in dictionary. ECG data may be with passes, so it must be checked by HFD method"""

        HFD_OF_ECG_1_AND_2 = {}
        AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES = {}
        ECG_COUNT_PER_EACH_AGE_GROUP = {}
        HIGUCHI_AVERAGE_PER_EACH_AGE_GROUP = {}
        AGE_INDEX_FOR_IDS_OF_BOTH_HFD_VALUES = {}

        SEXES_FOR_IDS_OF_BOTH_HFD_VALUES = {}
        BMIS_FOR_IDS_OF_BOTH_HFD_VALUES = {}
        LENGTH_FOR_IDS_OF_BOTH_HFD_VALUES = {}


        if (one_minute_pass):
            minute_points_from_ECG_start = 60000
        else:
            minute_points_from_ECG_start = 0

        with open(path+'/'+ csv_info_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)} ')
                    line_count += 1
                else:
                    # Check if id and age_group is not NaN

                    if(row[0]>'0100'):
                        continue

                    if (row[0] != 'NaN' and row[1] != 'NaN'):
                        print(f'\tId: {row[0]}; Age_group: {age_groups[row[1]]}; Sex: {gender[row[2]]}; BMI: {row[3]}; Length: {row[4]}; Device: {device[row[5]]}.')

                        length = find_length_of_record(row[0])

                        print("Points in ECG: " + str(length))

                        record = None

                        """Check, if ECG length > 1 min"""
                        if (one_minute_pass and length <= minute_points_from_ECG_start):
                            line_count += 1
                            continue

                        ### Select method of cut of ECG ###

                        ### Warning !!! For ECG length more than 1 min

                        if (cut_method == TypeOfECGCut.full):

                            record = open_record(row[0], minute_points_from_ECG_start, None)

                        if (cut_method == TypeOfECGCut.start):
                            record = open_record(row[0], minute_points_from_ECG_start, standart_length)

                        if (cut_method == TypeOfECGCut.end):

                            delta = 0

                            if ((length - standart_length) >= minute_points_from_ECG_start):
                                pass
                            else:
                                delta = length - standart_length - minute_points_from_ECG_start

                            record = open_record(row[0], length - standart_length - delta, None)

                        if (cut_method == TypeOfECGCut.middle):

                            # Test this case for reliable situation.
                            # If left_length and right length is different cut windows is translated left on 1 point than right
                            left_length = (length - minute_points_from_ECG_start  - standart_length) // 2

                            if (left_length >= 0):

                                record = open_record(row[0], minute_points_from_ECG_start + left_length, minute_points_from_ECG_start + left_length + standart_length)
                            else:
                                record = open_record(row[0], minute_points_from_ECG_start, standart_length)


                        if not isinstance(record, list):

                            continue

                        ################################################
                        #ecg = RECORD(row[0], row[1], row[2], row[3], row[4], row[5], record[0], record[1])

                        #if(row[0]<'0162'):

                        #old_ecg_dictionary[row[0]] = ecg

                        result = calculate_higuchi(record[0], record[1])


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


                    line_count += 1



        #RECORD.DATABASE[cut_method] = old_ecg_dictionary

        print(HFD_OF_ECG_1_AND_2)
        write_HFD_calculated_values_to_csv(HFD_OF_ECG_1_AND_2, AGE_INDEX_FOR_IDS_OF_BOTH_HFD_VALUES, AGE_RANGES_FOR_IDS_OF_BOTH_HFD_VALUES, cut_method, SEXES_FOR_IDS_OF_BOTH_HFD_VALUES,BMIS_FOR_IDS_OF_BOTH_HFD_VALUES,LENGTH_FOR_IDS_OF_BOTH_HFD_VALUES)



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
        #print(dictionary_HFD_ECG_1_2)"""

        #RECORD.print_database()


def write_HFD_calculated_values_to_csv(hfd_of_ecg_1_and_2, age_indexes_for_id, age_ranges_for_id, type_of_ecg_cut, sexes, bmis, length, ):
    # ECG 1 and 2 simulationusly

    with open('HFD_calculated_after_one_minute_' + type_of_ecg_cut.name + '_cut.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in hfd_of_ecg_1_and_2.keys():
            spamwriter.writerow([key, age_indexes_for_id[key], age_ranges_for_id[key], localize_floats(hfd_of_ecg_1_and_2[key][0]),
                                 localize_floats(hfd_of_ecg_1_and_2[key][1]), sexes[key], bmis[key], length[key]])

        """, RECORD.DATABASE[type_of_ecg_cut][key].Sex,
                                 RECORD.DATABASE[type_of_ecg_cut][key].BMI]"""
def write_number_of_ECGs_per_age_range_for_both_HFD(ecg_count_per_each_age_group, type_of_ecg_cut):
    with open('number_of_ECGs_per_each_age_range_' + type_of_ecg_cut.name + '_cut.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in ecg_count_per_each_age_group.keys():
            spamwriter.writerow([key, ecg_count_per_each_age_group[key]])

def write_average_HFD_values_for_each_age_range(higuchi_average_per_each_age_group, type_of_ecg_cut):
    with open('HFD_average_of_ECG_per_age_range_' + type_of_ecg_cut.name + '_cut.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for key in higuchi_average_per_each_age_group.keys():
            spamwriter.writerow([key, localize_floats(higuchi_average_per_each_age_group[key][0]),
                                     localize_floats(higuchi_average_per_each_age_group[key][1])])
def open_record(id, min_point, max_point):

    """ Open each record with ECG by Id

        Input parapeters:
            - Id - id of record
            - min_point - minimum point, at which starts ECG (including this point)
            - max_point - maximum point, at which ends ECG (not including this point)"""

    # wfdb.rdrecord(... [0, 1] - first two channels (ECG 1, ECG 2); [0] - only first ECG
    # 0 - The starting sample number to read for all channels (point from what graphic starts (min_point)).
    # None - The sample number at which to stop reading for all channels (max_point). Reads the entire duration by default.

    try:
        if (load_data_online == True):
            record = wfdb.rdrecord(
                id, min_point, max_point, [0, 1], pn_dir='autonomic-aging-cardiovascular/1.0.0/')
        else:
            record = wfdb.rdrecord(
                path + '/' + id, min_point, max_point, [0, 1])

    except:
        return math.nan

    wfdb.plot_wfdb(record, title='Record' + id + ' from Physionet Autonomic ECG')
    #display(record.__dict__)


    sequence_1 = []
    sequence_2 = []


    # print(record.p_signal)

    for x in record.p_signal:

        # Use first ECG
        sequence_1.append(x[0])

        # Use second ECG
        sequence_2.append(x[1])

    print("Length with one minute: " + str(len(sequence_1)))
    #print(sequence)

    return [sequence_1, sequence_2]


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

                    length = find_length_of_record(row[0])

                    if(not (math.isnan(length))):
                        if(length < min_length):
                            min_length = length
                            id = line_count


                line_count += 1


    print(id)

    # result on dataset 480501
    return min_length

def find_average_length_of_records():
    """Find minimum length of ECG record among all dataset"""

    min_length = 0
    id = 1
    """
    with open(path + '/' + csv_info_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # Check if id and age_group is not NaN
                if (row[0] != 'NaN' and row[1] != 'NaN'):

                    length = find_length_of_record(row[0])

                    if(not (math.isnan(length))):
                        if(length < min_length):
                            min_length = length
                            id = line_count


                line_count += 1
    """

    print(id)

    # result on dataset 480501 ... XXXX
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
                if (not (row[1] == 'NaN')):

                    length = find_length_of_record(row[0])

                    if(not (math.isnan(length))):
                        if(length > max_length):
                            max_length = length


                line_count += 1


    return max_length


def find_length_of_record(id):
    try:
        if (load_data_online == True):
            record = wfdb.rdrecord(
                id, 0, None, [0, 1], pn_dir='autonomic-aging-cardiovascular/1.0.0/')
        else:
            record = wfdb.rdrecord(
                path + '/' + id, 0, None, [0, 1])
    except:
        return math.nan


    return len(record.p_signal)




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




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



    #minimum_length = find_minimum_length_of_records()
    minimum_length = 480501

    #open_record('0637', 0, 480501)
    #number_of_ECG_by_each_age_group()
    read_ECG_data(minimum_length, TypeOfECGCut.full, True)


    ################################################################################################################





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
