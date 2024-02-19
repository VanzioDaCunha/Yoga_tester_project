"""
Created by Vanzio, Divya on 11/02/2024
last modified: 12/02/2024 10:40
"""


import csv


# function takes annotations csv file as input
# returns list containing the start time, end time and the activity
def read_csv_file(filename):
    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as file:
        # creating a csv reader object
        csv_reader = csv.reader(file)

        # extracting field names through first row
        # fields is used only to remove the header
        fields = next(csv_reader)

        # extracting each data row one by one
        for row in csv_reader:
            col = []
            for i, cols in enumerate(row):
                if i > 1 and cols != '':
                    col.append(cols)
            rows.append(col)

    return rows


# Function takes annotations as a list and the current time
# Returns value of the current activity during that particular time
def get_time(data, current_time):
    val = dict()
    for i in data:
        if float(i[0]) <= current_time <= float(i[1]):
            val = eval(i[2])

    # to check if dictionary is empty
    if val.__len__() == 0:
        return ''
    key = val['TEMPORAL-SEGMENTS']

    return key

