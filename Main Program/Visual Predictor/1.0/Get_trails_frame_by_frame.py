#!/usr/bin/python3

'''

Extract all frames from a video/multiple videos and the centroid of the elements in each frame (OpenCV with Python)
The filters to preprocess the images are: To HSV color, to gray scale, then to binary mode, erode, and then dilate.
After that we obtain the moments and the centroid of the entitys, in this case it was use to localize the center of a ball rolling.
Exports are Original with the centroid and dilated also with the centroid.


'''

__author__ = "Alvaro Martin"
__date__ = "07/11/2021"

import os
import cv2
import numpy as np
import Network, Preprocessing

def Extract_Frames(video):

    data_path = "/Users/Martin/Desktop/Generator_10/Frames_dataset/linear_point_255_fix_2000_80_120_30GAP/linear_30_[None]_test"
    model_path = "/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/REC/Frames_dataset/linear_point_255_fix_2000_80_120_30GAP_Modeled/simple/10_False_tanh_mean_squared_error_10.h5"

    data_type = data_path.split('/')[6]
    print(data_type)
    net_type = model_path.split('/')[9]
    print(net_type)
    complexity = model_path.split('/')[12]
    if "modeled" in model_path.lower():
        data_path = data_path + "/modeled_samples"
    sample_type = data_path.split('/')[-1]
    data_type = data_type + "_" + sample_type
    print(data_type)
    samples_dir = data_path.split('/')[6]
    print(samples_dir)
    dim = (int(samples_dir.split('_')[-3]), int(samples_dir.split('_')[-2]))
    print(dim)


    print('\n')
    print("Model: " + model_path)
    print('\n')
    print("Evaluating with " + data_type + " a " + complexity + " " + net_type + " model")
    print('Puting the test data into the right shape...')
    to_test_net = Network.Lstm(model_file=model_path, framework="tensorflow")

    gap = 30
    v = 0
    cX = 0
    previous_cX = 0
    max_number_frames = 19
    gap = max_number_frames + 30

    # List all the videos
    if video.endswith(".MP4"):
        FIRST_data = []
        GAP_data = []
        FINAL = []
        dataX = []
        video_name = video.split('/')[-1]
        video_path = video
        init = 0
        buffer = 20
        img_index = 0
        v += 1

        frames_path = video + '_frames'
        os.makedirs(frames_path,exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        os.chdir(frames_path)

        if len(dataX) != 0:
            FIRST_data.append(dataX)

        print('\nVideo #' + str(v) + '-----------' + str(video_name) + '\n')

        # List all the frames
        while (cap.isOpened()):

            ret, frame = cap.read()
            if ret == False:
                break

            def resize(img, width, height):

                dim = (width, height)
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

                return resized

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

            if int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0):

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                ok = []
                ko = []
                data_temp_x = []
                data_temp_y = []
                real_points = []
                predicted_points = []

                if (cX >= previous_cX):

                    if (img_index <= max_number_frames + 30):
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
                        ko.append(data_temp_y)
                        ko = np.array(ko)

                        print('Input numero ----- '+ str(init+1))
                        real_points, predicted_points = to_test_net.test(ok, ko, gap, data_type, dim)

                        print('BUFFER ----'+ str(init+1))
                        print(np.array(ok))

                        print('REAL ----- COMPROBANDO'+ str(init+1))
                        print(np.array(real_points))

                        print('PREDICTED  ----- REDONDEADO '+ str(init+1))
                        print(np.array(predicted_points))
                        print('\n')
                        FINAL.append(predicted_points)

                        init += 1
                        buffer += 1

                    else:
                        pass

                    if (img_index < gap + 20):

                        fps = cap.get(cv2.CAP_PROP_FPS)
                        fps = int(fps)
                        fps = str(fps)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # putting the FPS count on the frame
                        #cv2.putText(frame, fps+"fps", (1, 20), font, 0.5, (100, 255, 0), 3, cv2.LINE_AA)

                        for j in dataX:
                            cv2.circle(frame, (int(j[1]), int(j[0])), 1, (230, 0, 115), 1)
                            cv2.imwrite('Interface'+ str(img_index+1) + '.png', frame)

                        for j in GAP_data:
                            cv2.circle(frame, (int(j[1]), int(j[0])), 1, (230, 0, 115), 1)
                            cv2.imwrite('Interface'+ str(img_index+1) + '.png', frame)

                        for j in FINAL:
                            cv2.circle(frame, (int(j[1]), int(j[0])), 1, (30, 30, 240), 1)
                            cv2.imwrite('Interface'+ str(img_index+1) + '.png', frame)

                        f = frames_path +'/Interface'+ str(img_index+1) + '.png'
                        im = cv2.imread(f)

                        # Custom window
                        cv2.namedWindow('See the trails', cv2.WINDOW_KEEPRATIO)
                        cv2.imshow('See the trails', im)
                        cv2.resizeWindow('See the trails', 900, 600)

                        # show one frame at a time
                        key = cv2.waitKey(0)
                        while key not in [ord('q'), ord('a'), ord('n')] :
                            key = cv2.waitKey(0)
                        #if key == ord('a'):
                        #    f = frames_path +'/Interface'+ str(img_index-1) + '.png'
                        #    im = cv2.imread(f)
                        #    cv2.imshow('See the trails', im)
                        #    img_index -= 1
                        if key == ord('n'):
                            f = frames_path +'/Interface'+ str(img_index+1) + '.png'
                            im = cv2.imread(f)
                            cv2.imshow('See the trails', im)

                        # Quit when 'q' is pressed
                        if key == ord('q'):
                            break
                img_index += 1
                #cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()
    return dataX, GAP_data, FINAL
