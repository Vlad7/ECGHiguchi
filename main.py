# This is an ECG Higuchi script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Age groups
import math
import pandas as pd
import openpyxl
import enum
from IPython.display import display
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv
import matplotlib.pyplot as plt
import os.path


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


<<<<<<< HEAD
=======
from IPython.display import display
import numpy as np
import wfdb
import HiguchiFractalDimension.hfd
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats

>>>>>>> a7e22fc04678f933cea1483c155fdcd1e8416900
#######################################################################################################################

# Path to dataset of ECG
# For future make loading from web database
path = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'

csv_info_file = 'subject-info.csv'

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
        self.ECG_1 = ecg_1
        self.ECG_2 = ecg_2

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


#######################################################################################################################
############################################## OPENING RECORDS ########################################################
#######################################################################################################################
def read_ECGs_annotation_data():

        """ Open csv info file, print header and information for each record. Then fill ECG DATABASE. """

        with open(path+'/'+ csv_info_file) as csv_file:
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

                # Take only one ecg
                #if (line_count > 1):
                #    return

                # If Id is not available
                if (row[0] == 'NaN'):
                    continue


                # If age category is available
                if (not (row[1] == 'NaN')):
                    # Open record returns ecg_1 and ecg_2
                    ecg_s = open_record(row[0], 0, None)

                    # Calling constructor for RECORD and automatically saving to DATABASE
                    record = RECORD(row[0], row[1], row[2], row[3], row[4], row[5], ecg_s[0], ecg_s[1])
                else:
                    continue

                #if (not (row[0] == 'NaN')):
                #   selected_records.append(row[0])




                    

def open_record(id, min_point, max_point):

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
    try:
        if min_point < 0:
            print("Too low minimal point of ECG! Now minimal point is 0!")
            min_point = 0

        record = wfdb.rdrecord(
            path + '/' + id, min_point, max_point, [0, 1])
    except :
        if os.path.isfile(path + '/' + id+'.hea') or os.path.isfile(path + '/' + id+'.dat'):
            max_point = None
            print("Too hight maximal point of ECG! Now maximal point is None!")

    if record is None:

        try:
            record = wfdb.rdrecord(
                path + '/' + id, min_point, max_point, [0, 1])
        except:
            print("File with record doesn't open!")
            return None

    wfdb.plot_wfdb(record, title='Record ' + id + ' from Physionet Autonomic ECG')

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

    return [sequence_1, sequence_2]

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

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


    
######################################################################################################################
######################################################################################################################
######################################################################################################################

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


def write_HFD_calculated_values_to_csv(hfd_of_ecg_1_and_2, age_indexes_for_id, age_ranges_for_id, type_of_ecg_cut, sexes, bmis, length, minutes_passed):
    # ECG 1 and 2 simulationusly

    with open('HFD_calculated_after_minutes_' + str(total_minutes_points_from_ECG_start) + "_" + type_of_ecg_cut.name + '_cut.csv', 'w', newline='') as csvfile:
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
<<<<<<< HEAD
=======
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
        record = wfdb.rdrecord(
            path + '/' + id, min_point, max_point, [0, 1])
    except:
        return math.nan





    #display(record.__dict__)
>>>>>>> a7e22fc04678f933cea1483c155fdcd1e8416900




<<<<<<< HEAD
=======
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

>>>>>>> a7e22fc04678f933cea1483c155fdcd1e8416900
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
    read_ECGs_annotation_data()

    # Example usage
    path = "/path/to/data"  # Update this path to the location of your data files
    record_id = "0001"
    min_point = 0
    max_point = None

    sequences = [DATABASE['0001'].ECG_1, DATABASE['0001'].ECG_2]
    if sequences is not math.nan:
        save_to_csv(record_id, sequences, f"ecg_{record_id}.csv")




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
