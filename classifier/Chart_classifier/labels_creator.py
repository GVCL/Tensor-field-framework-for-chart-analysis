import glob
import os.path
from os import path
import sys
import csv
import pandas as pd
import cv2

def readFileNames(folder, pattern):
    return [image_path
            for x in os.walk(folder)
            for image_path in glob.glob(os.path.join(x[0], pattern))]

def write_to_csv(row):
    with open('labels.csv', 'a+') as my_csv:
        writer = csv.writer(my_csv)
        writer.writerow(row)

def main():
    file_path = sys.argv[1]
    chart_type = file_path.split('/')[-1]
    print (file_path, chart_type)
    csv_header = ['Image_path', 'Class']
    #### check if file exist if not do this
    if (not(path.exists('labels.csv'))):
        print ('here')
        with open('labels.csv', 'w+') as my_csv:
            writer = csv.writer(my_csv)
            writer.writerow(csv_header)
    image_list = readFileNames(file_path, '*')
    for image in image_list:
        row = [image, chart_type]
        write_to_csv(row)


if __name__ == '__main__':
    main()

def load_data(csv_path):
    image_data = []
    df = pd.read_csv(csv_path, sep=",", index_col=False)
    data_len = len(df['Image_path'])
    for i in range(data_len):
        temp = cv2.imread(df['Image_path'][i])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        image_data.append(temp, df['Class'][i])
    # return zip(df['Image_path'], df['Class'])
