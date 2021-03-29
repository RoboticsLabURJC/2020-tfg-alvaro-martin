"""

TFM - main_test.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"

import sys
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test')
#sys.path.insert(0, 'C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/')

from Utils import utils, func_utils, vect_utils, frame_utils
from Network import Net

if __name__ == '__main__':
    conf = utils.get_config_file()
    data_type = conf['data_path'].split('/')[6]
    print(data_type)
    net_type = conf['model_path'].split('/')[9]
    print(net_type)
    complexity = conf['model_path'].split('/')[12]

    print('\n')
    print("Dataset: " + conf['data_path'])
    print('\n')
    print("Model: " + conf['model_path'])
    print('\n')

    print("Evaluating with " + data_type + " a " + complexity + " " + net_type + " model")

    # Load data
  # data_type == "Frames_dataset
    if "modeled" in conf['model_path'].lower():
        data_path = conf['data_path'] + "/modeled_samples"
    else:
        data_path = conf['data_path'] + "/raw_samples"
    print(data_path)
    print('\n')
    sample_type = data_path.split('/')[-1]
    print(sample_type)
    data_type = data_type + "_" + sample_type
    print(data_type)
    samples_dir = data_path.split('/')[6]
    print(samples_dir)
    dim = (int(samples_dir.split('_')[-3]), int(samples_dir.split('_')[-2]))
    print(dim)
    if sample_type == "raw_samples":
        gauss_pixel = "gauss" in conf['model_path'].lower()
        print("Gauss:", gauss_pixel)
        if net_type == "NOREC":
            print('Puting the test data into the right shape...')
            parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, gauss_pixel)
            to_test_net = Net.Convolution2D(model_file=conf['model_path'], framework="keras")
        else:
            print('Puting the test data into the right shape...')
            parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, gauss_pixel, True)
            to_test_net = Net.ConvolutionLstm(model_file=conf['model_path'], framework="keras")
    else:
        parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, False)
        if net_type == "NOREC":
            print('Puting the test data into the right shape...')
            to_test_net = Net.Mlp(model_file=conf['model_path'], framework="tensorflow")
        else:
            print('Puting the test data into the right shape...')
            to_test_net = Net.Lstm(model_file=conf['model_path'], framework="tensorflow")

    gap = parameters.iloc[0]['gap']

    to_test_net.test(testX, testY, gap, data_type, dim)
