''' TEST NEW '''

import tensorflow as tf
import numpy as np
from time import time
import cv2
import sys
import os


np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test')
#sys.path.insert(0, 'C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/')
from Utils import utils, frame_utils, test_utils
#from Network import Net

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, LSTM, ConvLSTM2D, TimeDistributed
from keras.utils import vis_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint



class Net(object):

    def __init__(self, net_type, **kwargs):
        self.net_type = net_type
        self.framework = kwargs['framework']
        print(self.framework)
        if 'model_file' in kwargs.keys():
            self.model_path = kwargs['model_file'][:kwargs['model_file'].rfind("/") + 1]
            if self.framework == "keras":
                self.model = load_model(kwargs['model_file'])
            else:
                self.model = tf.keras.models.load_model(kwargs['model_file'])
        else:
            if self.framework == "keras":
                self.model = Sequential()
            else:
                self.model = tf.keras.Sequential()

            self.dropout = kwargs['dropout']
            if self.dropout:
                self.drop_percentage = kwargs['drop_percentage']
            self.loss = kwargs['loss']
            self.activation = kwargs['activation']
            self.input_shape = kwargs['input_shape']
            self.output_shape = kwargs['output_shape']



    def test(self, test_x, test_y, gap, data_type, dim):
        predict = self.model.predict(test_x)
        with open('predict.txt', 'w') as file:
                file.write(str(predict))

        raw = True
        if "modeled" in data_type:
            raw = False
        predict_values, real_values, maximum = frame_utils.get_positions(predict, test_y, dim, raw)

        if raw:
            v_to_draw = predict
        else:
            v_to_draw = predict_values

        error, x_error, y_error, relative_error = test_utils.calculate_error(real_values, predict_values, maximum)

        with open(self.model_path + 'error_result.txt', 'w') as file:
            for i in range(error.shape[0]):
                file.write("Processed sample " + str(i) + ": \n")
                file.write("Target position: " + str(real_values[i]) + "\n")
                file.write("Position: " + str(predict_values[i]) + "\n")
                file.write("Error: " + str(np.round(error[i], 2)) + " (" + str(np.round(relative_error[i], 2)) + "%)\n")
                file.write("--------------------------------------------------------------\n")

                real_x = int(real_values[i][0])
                real_y = int(real_values[i][1])
                pr_x = int(predict_values[i][0])
                pr_y = int(predict_values[i][1])

                w, h = 120, 80
                black = (0, 0, 0)
                black_img = np.zeros((h, w, 3), np.uint8)
                black_img_2 = np.zeros((h, w, 3), np.uint8)

                # Since OpenCV uses BGR, convert the color first
                color = tuple(reversed(black))
                # Fill image with color
                black_img[:] = color
                black_img_2[:] = color

                folder_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/REC/Frames_dataset/linear_point_255_fix_1000_80_120_Modeled/simple/'
                real_path = folder_path + '/_real_trails'
                os.makedirs(real_path,exist_ok=True)
                predict_path = folder_path + '/_predicted_trails'
                os.makedirs(predict_path,exist_ok=True)

                os.chdir(real_path)
                cv2.circle(black_img, (real_y, real_x), 1, (246, 209, 81), -1)
                cv2.imwrite('Real_Trails' + str(i) + '.png', black_img)
                os.chdir(predict_path)
                cv2.circle(black_img_2, (pr_y, pr_x), 1, (15, 232, 253), -1)
                cv2.imwrite('Real_Trails' + str(i) + '.png', black_img_2)

        # Calculate stats
        test_utils.get_error_stats(test_x, test_y, v_to_draw, gap, data_type, dim,
                                   error, x_error, y_error, relative_error, self.model_path)

class Lstm(Net):
    def __init__(self, **kwargs):
        Net.__init__(self, "lstm", **kwargs)
        if 'model_file' not in kwargs.keys():
            if kwargs['data_type'] == "Vector":
                self.create_vector_model()
            else:  # kwargs['data_type'] == "Frame"
                if kwargs['complexity'] == "simple":
                    self.create_frame_simple_model()
                else:
                    self.create_frame_complex_model()

    def create_vector_model(self):
        print("Creating function LSTM model")
        self.model.add(LSTM(25, input_shape=self.input_shape))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_frame_simple_model(self):
        print("Creating frame simple LSTM model")
        self.model.add(LSTM(25, input_shape=self.input_shape))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_frame_complex_model(self):
        print("Creating frame complex LSTM model")

        self.model.add(LSTM(70, return_sequences=True,  input_shape=self.input_shape))
        self.model.add(LSTM(40, return_sequences=True))
        self.model.add(LSTM(25, return_sequences=True))
        self.model.add(LSTM(15, return_sequences=False))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')


if __name__ == '__main__':
    data_path = '/Users/Martin/Desktop/Generator_10/Frames_dataset/linear_point_255_fix_1000_80_120_30GAP/linear_30_[None]_test'
    model_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/REC/Frames_dataset/linear_point_255_fix_1000_80_120_Modeled_30GAP/simple/10_False_tanh_mean_squared_error_10.h5'

    data_type = data_path.split('/')[6]
    net_type = model_path.split('/')[7]
    complexity = model_path.split('/')[10]

    print('\n')
    print("Dataset: " + data_path)
    print('\n')
    print("Model: " + model_path)
    print('\n')

    print("Evaluating with " + data_type + " a " + complexity + " " + net_type + " model")

    if "modeled" in model_path.lower():
        data_path = data_path + "/modeled_samples"
    else:
        data_path = data_path + "/raw_samples"
    print(data_path)
    print('\n')
    sample_type = data_path.split('/')[-1]
    print(sample_type)
    data_type = data_type + "_" + sample_type
    samples_dir = data_path.split('/')[6]
    dim = (int(samples_dir.split('_')[-2]), int(samples_dir.split('_')[-2]))

    if sample_type == "raw_samples":
        gauss_pixel = "gauss" in model_path.lower()
        print("Gauss:", gauss_pixel)
        if net_type == "NOREC":
            print('Puting the test data into the right shape...')
            parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, gauss_pixel)
            to_test_net = Convolution2D(model_file=model_path, framework="keras")
        else:
            print('Puting the test data into the right shape...')
            parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, gauss_pixel, True)
            to_test_net = ConvolutionLstm(model_file=model_path, framework="keras")
    else:
        parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, False)
        with open('testX.txt', 'w') as file:
                file.write(str(testX))
                print(str(testX))
        with open('testY.txt', 'w') as file:
                file.write(str(testY))
                print(str(testY))
        if net_type == "NOREC":
            print('Puting the test data into the right shape...')
            to_test_net = Mlp(model_file=model_path, framework="tensorflow")
        else:
            print('Puting the test data into the right shape...')
            to_test_net = Lstm(model_file=model_path, framework="tensorflow")
            with open('net.txt', 'w') as file:
                    file.write(str(to_test_net))

    gap = parameters.iloc[0]['gap']
    print(str(gap))

    to_test_net.test(testX, testY, gap, data_type, dim)
