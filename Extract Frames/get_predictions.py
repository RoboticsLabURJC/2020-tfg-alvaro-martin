import tensorflow as tf
import numpy as np
from time import time
import cv2
import sys
import os


np.set_printoptions(threshold=sys.maxsize)

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, LSTM, ConvLSTM2D, TimeDistributed
from keras.utils import vis_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint# File path

filepath = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/REC/Frames_dataset/linear_point_255_fix_1000_80_120_Modeled_30GAP/simple/10_False_tanh_mean_squared_error_10.h5'

# Load the model
model = load_model(filepath, compile = True)

# A few random samples
use_samples = [[[38.0, 1.0], [37.0, 1.0], [37.0, 2.0], [37.0, 2.0], [36.0, 3.0], [36.0, 3.0], [36.0, 4.0], [36.0, 5.0], [36.0, 6.0], [35.0, 7.0], [35.0, 7.0], [35.0, 8.0], [35.0, 10.0], [35.0, 11.0], [35.0, 13.0], [35.0, 14.0], [35.0, 15.0], [35.0, 17.0], [34.0, 18.0], [34.0, 19.0], [34.0, 21.0], [34.0, 22.0], [34.0, 24.0], [34.0, 25.0], [34.0, 26.0], [33.0, 28.0]], [[34.0, 1.0], [33.0, 2.0], [33.0, 2.0], [33.0, 3.0], [33.0, 4.0], [33.0, 5.0], [33.0, 5.0], [32.0, 6.0], [32.0, 7.0], [32.0, 8.0], [32.0, 10.0], [32.0, 11.0], [32.0, 13.0], [32.0, 15.0], [32.0, 16.0], [31.0, 18.0], [31.0, 20.0], [31.0, 21.0], [31.0, 23.0], [31.0, 24.0], [31.0, 26.0], [31.0, 27.0], [31.0, 29.0], [31.0, 31.0], [31.0, 32.0], [30.0, 34.0]]]
samples_to_predict = [[31.0, 65.0], [31.0, 66.0], [31.0, 67.0], [31.0, 68.0], [31.0, 69.0], [31.0, 71.0], [30.0, 72.0], [30.0, 73.0], [30.0, 74.0], [30.0, 75.0], [30.0, 76.0], [30.0, 77.0], [30.0, 78.0], [30.0, 79.0], [30.0, 80.0], [30.0, 82.0], [30.0, 83.0], [30.0, 84.0], [29.0, 85.0], [29.0, 86.0], [29.0, 87.0], [29.0, 76.0], [29.0, 78.0], [29.0, 79.0], [29.0, 81.0], [29.0, 82.0], [29.0, 83.0], [29.0, 85.0], [29.0, 86.0], [29.0, 87.0], [29.0, 88.0], [29.0, 90.0], [29.0, 91.0], [29.0, 92.0], [29.0, 94.0], [29.0, 95.0], [30.0, 96.0], [30.0, 97.0], [30.0, 99.0], [30.0, 100.0], [30.0, 101.0], [30.0, 103.0]]

# Convert into Numpy array
samples_to_predict = np.array(samples_to_predict)

# Generate predictions for samples
predictions = model.predict(samples_to_predict)
print(predictions)
