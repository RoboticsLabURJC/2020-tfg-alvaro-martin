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

"""Create new image filled with certain color in RGB"""
def create_blank(width, height, rgb_color=(0, 0, 0)):

    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

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


def Extract_Frames():

    v = 0
    cX = 0
    previous_cX = 0

    max_number_frames = 20
    gap = max_number_frames + 30

    # Create new blank image
    w, h = 120, 80
    #w, h = 120, 80
    black = (0, 0, 0)

    testX = []
    dataX = []
    testY = []

    # List all the videos
    for video in os.listdir(folder_path):
        if video.endswith(".MP4"):
            video_name = video
            video_path = folder_path + video
            img_index = 1
            v += 1

            frames_path = folder_path + '/' + video_name + '_frames'
            os.makedirs(frames_path,exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            os.chdir(frames_path)
            black_img = create_blank(w, h, rgb_color=black)


            if len(dataX) != 0:
                testX.append(dataX)

            dataX = []

            print('\nVideo #' + str(v) + '-----------' + str(video_name) + '\n')

            # List all the frames
            while (cap.isOpened()):

                ret, frame = cap.read()
                if ret == False:
                    break

                ''' FRAME RESIZED '''
                frame = resize(frame, 120, 80)
                ''' FRAME RESIZED '''

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.bitwise_not(hsv) # Inverted filter color

                # Threshold in HSV space
                lower = np.array([225, 0, 0]) # Orange
                #lower = np.array([0, 0, 160]) # Green Background
                #lower = np.array([0, 0, 160]) # Black background
                upper = np.array([255, 255, 255])

                # The black region in the mask has the value of 0
                mask = cv2.inRange(hsv, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

                # Gray scale
                gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                # Binary Mode
                ret,binary_gray = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)

                kernel = np.ones((3,3), np.uint8)       # 3x3 matrix
                erosion_image = cv2.erode(binary_gray, kernel, iterations=1)

                kernel = np.ones((5,5), np.uint8)     # 10x10 matrix
                dilation_image = cv2.dilate(erosion_image, kernel, iterations=1)

                # calculate moments of binary image
                M = cv2.moments(dilation_image)
                previous_cX = cX

                if int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0):

                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    data_temp_x = []
                    data_temp_y = []

                    if img_index <= max_number_frames:
                        data_temp_x.append(np.array(cY))
                        data_temp_x.append(np.array(cX))
                        dataX.append(data_temp_x)

                        cv2.circle(dilation_image, (cX, cY), 3, (0, 0, 0), -1)
                        #cv2.putText(dilation_image, "here", (cX - 10, cY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.circle(frame, (cX, cY), 3, (0, 0, 0), -1)
                        #cv2.putText(frame, "here", (cX - 10, cY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        if (cX >= previous_cX):
                            print ('Frame#' + str(img_index) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
                            cv2.imwrite('HVS+GREY+BIN(ERODE+DILATE) ' + str(img_index) + '.png', dilation_image)
                            cv2.imwrite('ORIGINAL ' + str(img_index) + '.png', frame)
                            cv2.circle(black_img, (cX, cY), 1, (246, 209, 81), -1)
                            cv2.imwrite('Real_Trails.png', black_img)

                    if (img_index == gap):
                        data_temp_y.append(np.array(cY))
                        data_temp_y.append(np.array(cX))
                        testY.append(data_temp_y)
                        print ('Frame#' + str(img_index) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
                        cv2.circle(black_img, (cX, cY), 1, (128, 0, 224), -1)
                        cv2.imwrite('Real_Trails.png', black_img)

                    if (img_index > gap) and (img_index <= gap + 20):

                        print ('Frame#' + str(img_index) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
                        cv2.circle(black_img, (cX, cY), 1, (15, 232, 253), -1)
                        cv2.imwrite('Real_Trails.png', black_img)

                    img_index += 1
                    cv2.waitKey(1)

                    '''
                    testX, testY = frame_utils.read_frame_data(data_path, sample_type, False)
                    to_test_net = Lstm(model_file=model_path, framework="tensorflow")

                    gap = 30
                    to_test_net.test(testX, testY, gap, data_type, dim)

                    predict = self.model.predict(test_x)
                    predict_values, real_values, maximum = frame_utils.get_positions(predict, test_y, dim, raw)

                    get predict_values, real_values



                    '''



    testX.append(dataX)
    cap.release()
    cv2.destroyAllWindows()

    return testX, testY

if __name__ == '__main__':
    folder_path = '/Users/Martin/Desktop/Nuevas tomas/'
    #folder_path = '/Users/Martin/Desktop/Prueba crudas pelota golf/Naranja/100 fps/'
    testX, testY = Extract_Frames()
    print("\n----- DONE. ALL IMAGES PROCESSED -----\n")
    os.chdir(folder_path)
    print('\n--- REAL VALUES 20 FIRST ---')
    with open('First 20.txt', 'w') as file:
            testX = (np.array(testX))
            file.write(str(testX))
            print(testX)

    print('\n--- GAP + 30 ---')
    with open('GAP + 30.txt', 'w') as file:
            testY = (np.array(testY))
            file.write(str(testY))
            print(testY)

    data_path = '/Users/Martin/Desktop/Generator_10/Frames_dataset/linear_point_255_fix_1000_80_120_30GAP/linear_30_[None]_test'
    model_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/REC/Frames_dataset/linear_point_255_fix_1000_80_120_Modeled_30GAP/simple/10_False_tanh_mean_squared_error_10.h5'

    data_type = data_path.split('/')[6]
    net_type = model_path.split('/')[7]
    complexity = model_path.split('/')[10]
    if "modeled" in model_path.lower():
        data_path = data_path + "/modeled_samples"
    sample_type = data_path.split('/')[-1]
    #print(sample_type)
    data_type = data_type + "_" + sample_type
    samples_dir = data_path.split('/')[6]
    dim = (int(samples_dir.split('_')[-2]), int(samples_dir.split('_')[-2]))

    #print('\n')
    #print("Dataset: " + data_path)
    print('\n')
    print("Model: " + model_path)
    print('\n')

    print("Evaluating with " + data_type + " a " + complexity + " " + net_type + " model")

    #if "modeled" in model_path.lower():
    #    data_path = data_path + "/modeled_samples"

    #parameters, testX, testY = frame_utils.read_frame_data(data_path, sample_type, False)

    print('Puting the test data into the right shape...')
    to_test_net = Lstm(model_file=model_path, framework="tensorflow")
    #with open('net.txt', 'w') as file:
    #        file.write(str(to_test_net))

    gap = 30
    #gap = parameters.iloc[0]['gap']
    #print(str(gap))
    print(data_type)
    print(dim)
    to_test_net.test(testX, testY, gap, data_type, dim)
