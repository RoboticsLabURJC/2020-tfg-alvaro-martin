'''

Extract all frames from a video/multiple videos and the centroid of the elements in each frame (OpenCV with Python)
The filters to preprocess the images are: To HSV color, to gray scale, then to binary mode, erode, and then dilate.
After that we obtain the moments and the centroid of the entitys, in this case it was use to localize the center of a ball rolling.
Exports are Original with the centroid and dilated also with the centroid.


'''

__author__ = "Alvaro Martin"
__date__ = "13/03/2021"

import os
import numpy as np
import cv2

import tensorflow as tf
import numpy as np
from time import time
import cv2
import sys
import os


np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0, '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test')
from Utils import utils, frame_utils, test_utils

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, LSTM, ConvLSTM2D, TimeDistributed
from keras.utils import vis_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint


''' Resize image '''
def resize(img, width, height):

    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized

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
        print(predict)

        raw = True
        if "modeled" in data_type:
            raw = False
        predict_values, real_values, maximum = frame_utils.get_positions(predict, test_y, dim, raw)

        predict_values = ([(int(predict_values[0][0])/2), (int(predict_values[0][1]*0.8))])

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


def Extract_Frames():

    v = 0
    cX = 0
    previous_cX = 0

    max_number_frames = 19
    gap = max_number_frames + 30

    FIRST_data = []
    dataX = []
    GAP_data = []
    FINAL = []

    # List all the videos
    for video in os.listdir(folder_path):
        if video.endswith(".MP4"):
            video_name = video
            video_path = folder_path + video
            init = 0
            buffer = 20
            img_index = 0
            v += 1

            frames_path = folder_path + '/' + video_name + '_frames'
            os.makedirs(frames_path,exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            os.chdir(frames_path)

            if len(dataX) != 0:
                FIRST_data.append(dataX)

            dataX = []

            print('\nVideo #' + str(v) + '-----------' + str(video_name) + '\n')

            # List all the frames
            while (cap.isOpened()):

                ret, frame = cap.read()
                if ret == False:
                    break

                ''' FRAME RESIZED '''
                frame = resize(frame, 120, 80)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.bitwise_not(hsv) # Inverted filter color

                # Threshold in HSV space
                lower = np.array([225, 0, 0]) # Orange
                upper = np.array([255, 255, 255])

                # The black region in the mask has the value of 0
                mask = cv2.inRange(hsv, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

                # Gray scale
                gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                # Binary Mode
                ret,binary_gray = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)

                kernel = np.ones((2,2), np.uint8)       # 3x3 matrix
                erosion_image = cv2.erode(binary_gray, kernel, iterations=1)

                kernel = np.ones((2,2), np.uint8)     # 10x10 matrix
                dilation_image = cv2.dilate(erosion_image, kernel, iterations=1)

                # calculate moments of binary image
                M = cv2.moments(dilation_image)
                previous_cX = cX

                real_points = []
                predicted_points = []

                if int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0):

                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    ok = []
                    ko = []
                    data_temp_x = []
                    data_temp_y = []
                    #FIRST_data = []

                    if (cX >= previous_cX):

                        if (img_index <= max_number_frames + 20):
                            print ('Frame#' + str(img_index+1) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
                            data_temp_x.append(np.array(cY))
                            data_temp_x.append(np.array(cX))
                            dataX.append(data_temp_x)

                        elif (img_index >= gap) and (img_index < gap + 20):
                            print ('Frame#' + str(img_index+1) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
                            data_temp_y.append(np.array(cY))
                            data_temp_y.append(np.array(cX))


                            FIRST_data.append(dataX)
                            GAP_data.append(data_temp_y)

                            ok.append(FIRST_data[0][init:buffer])
                            ok = np.array(ok)
                            print(ok)
                            ko.append(data_temp_y)
                            ko = np.array(ko)
                            print(ko)

                            print('Input numero ----- '+ str(init+1))
                            real_points, predicted_points = to_test_net.test(ok, ko, gap, data_type, dim)

                            print('BUFFER ----'+ str(init+1))
                            print(np.array(ok))
                            print('input + 30 frames gap ----- '+ str(init+1))
                            print(np.array(ko))

                            print('REAL ----- '+ str(init+1))
                            print(np.array(real_points))

                            print('PREDICTED ----- '+ str(init+1))
                            print(np.array(predicted_points))
                            #cv2.circle(frame, (int(real_points[0][1]), int(real_points[0][0])), 1, (246, 209, 81), 1)
                            #cv2.circle(frame, (int(predicted_points[1]), int(predicted_points[0])), 1, (0, 0, 0), 1)
                            #cv2.imwrite('Interface'+ str(init+50) + '.png', frame)
                            FINAL.append(predicted_points)
                            #for j in dataX:
                            #        cv2.circle(frame, (int(j[1]), int(j[0])), 1, (246, 209, 81), 1)
                            #        cv2.imwrite('Interface'+ str(img_index+1) + '.png', frame)

                            init += 1
                            buffer += 1

                        else:
                            pass

                    if (img_index < gap + 20):

                        for j in dataX:
                                cv2.circle(frame, (int(j[1]), int(j[0])), 1, (246, 209, 81), 1)
                                #cv2.circle(frame, (int(FINAL[j][1]), int(FINAL[j][0])), 1, (0, 0, 0), 1)
                                cv2.imwrite('Interface'+ str(img_index+1) + '.png', frame)

                        for j in GAP_data:
                                cv2.circle(frame, (int(j[1]), int(j[0])), 1, (246, 209, 81), 1)
                                cv2.imwrite('Interface'+ str(img_index+1) + '.png', frame)

                        for j in FINAL:
                                cv2.circle(frame, (int(j[1]), int(j[0])), 1, (0, 0, 0), 1)
                                cv2.imwrite('Interface'+ str(img_index+1) + '.png', frame)

                        f = '/Users/Martin/Desktop/Nuevas tomas/Prueba Interfaz/P1070380.MP4_frames/Interface'+ str(img_index+1) + '.png'
                        im = cv2.imread(f)
                        # Custom window
                        cv2.namedWindow('See the trails', cv2.WINDOW_KEEPRATIO)
                        cv2.imshow('See the trails', im)
                        cv2.resizeWindow('See the trails', 600, 400)

                    img_index += 1
                    cv2.waitKey(300)


    #
    cap.release()
    cv2.destroyAllWindows()

    #return np.array(FIRST_data), np.array(GAP_data)


if __name__ == '__main__':
    folder_path = '/Users/Martin/Desktop/Nuevas tomas/Prueba Interfaz/'

    data_path = '/Users/Martin/Desktop/Generator_10/Frames_dataset/linear_point_255_fix_1000_80_120_30GAP/linear_30_[None]_test'
    model_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/REC/Frames_dataset/linear_point_255_fix_1000_80_120_Modeled_30GAP/simple/10_False_tanh_mean_squared_error_10.h5'

    data_type = data_path.split('/')[6]
    net_type = model_path.split('/')[7]
    complexity = model_path.split('/')[10]
    if "modeled" in model_path.lower():
        data_path = data_path + "/modeled_samples"
    sample_type = data_path.split('/')[-1]
    data_type = data_type + "_" + sample_type
    samples_dir = data_path.split('/')[6]
    dim = (int(samples_dir.split('_')[-3]), int(samples_dir.split('_')[-2]))


    print('\n')
    print("Model: " + model_path)
    print('\n')
    print("Evaluating with " + data_type + " a " + complexity + " " + net_type + " model")
    print('Puting the test data into the right shape...')
    to_test_net = Lstm(model_file=model_path, framework="tensorflow")

    gap = 30

    #FIRST_data, GAP_data = Extract_Frames()

    Extract_Frames()
    '''
    init = 0
    buffer = 20
    frames_to_predict = 19
    real_points = []
    predicted_points = []

    for j in FIRST_data[0]:
        if init <= frames_to_predict:
            ok = []
            ko = []
            ok.append(FIRST_data[0][init:buffer])
            ok = (np.array(ok))
            ko.append(GAP_data[init])
            ko = (np.array(ko))

            print('Input numero ----- '+ str(init+1))
            real_points, predicted_points = to_test_net.test(ok, ko, gap, data_type, dim)

            print('BUFFER ----'+ str(init+1))
            print(np.array(ok))
            print('input + 30 frames gap ----- '+ str(init+1))
            print(np.array(ko))

            print('REAL ----- '+ str(init+1))
            print(np.array(real_points))

            print('PREDICTED ----- '+ str(init+1))
            print(np.array(predicted_points))

            f = '/Users/Martin/Desktop/Nuevas tomas/Prueba Interfaz/P1070380.MP4_frames/Interface'+ str(init+50) + '.png'
            im = cv2.imread(f)
            cv2.circle(im, (int(real_points[0][1]), int(real_points[0][0])), 1, (246, 209, 81), 1)
            cv2.imwrite('Interface'+ str(init+50) + '.png', im)
            cv2.circle(im, (int(predicted_points[1]), int(predicted_points[0])), 1, (0, 0, 0), 1)
            cv2.imwrite('Interface'+ str(init+50) + '.png', im)
            # Custom window
            cv2.namedWindow('See the trails', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('See the trails', im)
            cv2.resizeWindow('See the trails', 600, 400)
            cv2.waitKey(150)

            init += 1
            buffer += 1
    '''
