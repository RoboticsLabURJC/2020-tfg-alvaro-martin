#!/usr/bin/python3

import sys
import numpy as np

sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test')
from Utils import utils, frame_utils, test_utils

def get_graphs(buffer, real_values, predict_values, maximum, folder_path, data_type, dim):
    error, x_error, y_error, relative_error = test_utils.calculate_error(real_values, predict_values, maximum)
    #print(error, x_error, y_error, relative_error)

    with open(folder_path + 'error_result.txt', 'w') as file:
        for i in range(error.shape[0]):
            file.write("Processed sample " + str(i) + ": \n")
            file.write("Target position: " + str(real_values[i]) + "\n")
            file.write("Position: " + str(predict_values[i]) + "\n")
            file.write("Error: " + str(np.round(error[i], 2)) + " (" + str(np.round(relative_error[i], 2)) + "%)\n")
            file.write("--------------------------------------------------------------\n")

    # Calculate stats
    test_utils.get_error_stats(buffer, real_values, predict_values, 30, data_type, dim,
                               error, x_error, y_error, relative_error, folder_path)
