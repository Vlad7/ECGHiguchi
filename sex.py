path_to_dataset_folder = 'D:/SCIENCE/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'
#path_to_dataset_folder  = 'C:/Datasets/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0'
csv_info_file = 'subject-info.csv'
# Age groups
import csv
import enum
import math
import numpy as np
import pandas as pd
import openpyxl

#from IPython.display import display

import wfdb
import HiguchiFractalDimension.hfd

import matplotlib.pyplot as plt
import os.path
import sys
import os

def get_sex_for_each_id(ids, is_remotely=False):
    """Get id's list for each sex

        input:
            is_remotely - load annotation file from the internet

        output:

            dictionary with id as key and sex as value
    """

    # Dictionary with id as key and sex as value
    sex_dictionary = {}

    # Check, if dataset is remotely located
    if is_remotely:
        path = csv_info_file
    else:
        path = path_to_dataset_folder + '/' + csv_info_file

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # Get first raw with attributes
        first_row = next(csv_reader)

        # Setting counter for first row
        line_count = 0

        # We are processing the remaining rows
        for row in csv_reader:
            line_count += 1

            # Row[0] - id of record, row[2] - sex in index form
            if row[0] in ids:
                sex_dictionary[row[0]] = row[2]

    return sex_dictionary

