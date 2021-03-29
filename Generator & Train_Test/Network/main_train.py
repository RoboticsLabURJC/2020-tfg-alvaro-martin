"""

TFM - main_train.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"

import sys
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/')
#sys.path.insert(0, 'C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/')


from Utils import utils, func_utils, vect_utils, frame_utils
from Network import Net

import numpy as np


if __name__ == '__main__':
    conf = utils.get_config_file()

    net_type = conf['net_type']
    activation = conf['activation']
    dropout = conf['dropout']['flag']
    drop_percentage = float(conf['dropout']['percentage'])
    n_epochs = conf['n_epochs']
    batch_size = conf['batch_size']
    patience = conf['patience']

    data_dir = conf['data_dir']
    data_type = data_dir.split('/')[5]
    func_type = data_dir.split('/')[6]

    root = conf['root'] + net_type.upper() + '/' + data_type + '/' + func_type
    print(root)
    version = conf['version']

    print('Puting the data into the right shape...')

    # data_type == 'Frames_dataset':
    print('Training with frames')
    data_model = conf['data_model']
    samples_dir = data_dir.split('/')[6]
    dim = (int(samples_dir.split('_')[-3]), int(samples_dir.split('_')[-2]))
    complexity = conf['complexity']

    # Load data
    channels = False
    if data_model == "raw":
        batch_data = conf['batch_data']
        loss = conf['raw_frame_loss']
        gauss_pixel = conf['gauss_pixel']

        print("Raw images")
        if net_type == "Rec":
            channels = True

        if gauss_pixel:
            filename = root + "_Gauss/" + complexity
        else:
            filename = root + "/" + complexity

        if batch_data:
            train_data = utils.get_dirs(data_dir + 'train/raw_samples')
            val_data = utils.get_dirs(data_dir + 'val/raw_samples')
            images_per_sample = frame_utils.get_images_per_sample(train_data[0])
            if channels:
                in_dim = [images_per_sample, dim[0], dim[1], 1]
            else:
                in_dim = [images_per_sample, dim[0], dim[1]]
        else:
            _, trainX, trainY = frame_utils.read_frame_data(data_dir + 'train/', 'raw_samples',
                                                            gauss_pixel, channels)
            _, valX, valY = frame_utils.read_frame_data(data_dir + 'val/', 'raw_samples', gauss_pixel, channels)
            train_data = [trainX, trainY]
            val_data = [valX, valY]
            in_dim = trainX.shape[1:]

        out_dim = np.prod(in_dim[1:])

        # Model settings
        if net_type == "NoRec":
            to_train_net = Net.Convolution2D(activation=activation, loss=loss, dropout=dropout,
                                             drop_percentage=drop_percentage, input_shape=in_dim,
                                             output_shape=out_dim, complexity=complexity, framework="keras")
        else:
            to_train_net = Net.ConvolutionLstm(activation=activation, loss=loss, dropout=dropout,
                                               drop_percentage=drop_percentage, input_shape=in_dim,
                                               output_shape=out_dim, complexity=complexity, framework="keras")

    else:
        print("Modeled images")
        batch_data = False
        loss = conf['modeled_frame_loss']
        gauss_pixel = False
        activation = conf['modeled_activation']
        dim = (int(samples_dir.split('_')[-3]), int(samples_dir.split('_')[-2]))
        filename = root + "_Modeled/" + complexity

        _, trainX, trainY = frame_utils.read_frame_data(data_dir + 'train/', 'modeled_samples', gauss_pixel)
        _, valX, valY = frame_utils.read_frame_data(data_dir + 'val/', 'modeled_samples', gauss_pixel)
        train_data = (trainX, trainY)
        print(train_data)
        val_data = (valX, valY)
        print(val_data)

        # Model settings
        in_dim = trainX.shape[1:]
        out_dim = np.prod(in_dim[1:])
        if net_type == "NoRec":
            to_train_net = Net.Mlp(activation=activation, loss=loss, dropout=dropout,
                                   drop_percentage=drop_percentage, input_shape=in_dim, output_shape=out_dim,
                                   complexity=complexity, data_type="Frame", framework="tensorflow")
        else:  # net_type == "Rec"
            to_train_net = Net.Lstm(activation=activation, loss=loss, dropout=dropout,
                                    drop_percentage=drop_percentage, input_shape=in_dim, output_shape=out_dim,
                                    complexity=complexity, data_type="Frame", framework="keras")

    print('Training')
    to_train_net.train(n_epochs, batch_size, patience, filename, train_data, val_data, batch_data, gauss_pixel, channels)
