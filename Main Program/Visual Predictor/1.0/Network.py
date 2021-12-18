#!/usr/bin/python3

'''

Extract all frames from a video/multiple videos and the centroid of the elements in each frame (OpenCV with Python)
The filters to preprocess the images are: To HSV color, to gray scale, then to binary mode, erode, and then dilate.
After that we obtain the moments and the centroid of the entitys, in this case it was use to localize the center of a ball rolling.
Exports are Original with the centroid and dilated also with the centroid.


'''

__author__ = "Alvaro Martin"
__date__ = "13/03/2021"

import numpy as np
import sys
import tensorflow as tf
from time import time

np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test')
from Utils import utils, frame_utils, test_utils



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
                print('Modelo Cargado')
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
        #print(predict)

        raw = True
        if "modeled" in data_type:
            raw = False
        predict_values, real_values, maximum = frame_utils.get_positions(predict, test_y, dim, raw)

        #predict_values = ([(int(predict_values[0][0])/2), (int(predict_values[0][1]*0.8))])
        predict_values = ([(round(int(predict_values[0][0]), 0)), (int(predict_values[0][1]))])

        return (real_values, predict_values)


class Lstm(Net):
    def __init__(self, **kwargs):
        Net.__init__(self, "lstm", **kwargs)
        if 'model_file' not in kwargs.keys():
            if kwargs['complexity'] == "simple":
                self.create_frame_simple_model()
            else:
                self.create_frame_complex_model()

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
