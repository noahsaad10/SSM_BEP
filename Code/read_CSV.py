import csv
import numpy as np
import glob

# Specify the file path of the CSV file
files = glob.glob('./COW/boven/landmarks/*.csv')


for j in range(len(files)):
    # Open the CSV file
    with open(files[j], 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        data = []
        # Read the CSV data into a list
        data_list = list(csv_reader)
        data_list = data_list[1:]
        for i in range(len(data_list)):
            data.append(data_list[i][1:])

    file_name = files[j][28:-4]
    # Convert the data list to a NumPy array
    data_array = np.array(data, dtype=  float)

    np.savetxt(f'./COW/boven/landmarks/{file_name}.txt', data_array)

