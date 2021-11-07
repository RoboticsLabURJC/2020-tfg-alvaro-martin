#!/usr/bin/python3

import os
import numpy as np
import cv2
#import Extract_centroid_live as ecl

def Video_Live_Capture():

    gap = 30
    v = 0
    cX = 0
    previous_cX = 0
    max_number_frames = 19
    gap = max_number_frames + 30
    FIRST_data = []
    GAP_data = []
    FINAL = []
    dataX = []
    init = 0
    buffer = 20
    img_index = 0
    v += 1

    frames_path = '/Users/Martin/Desktop/TEST_LIVE'
    os.makedirs(frames_path,exist_ok=True)
    os.chdir(frames_path)

    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        #cv2.imshow('Visor', frame)
        # ecl.Extract_Frames_Live()

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
        lower = np.array([200, 0, 0]) # Orange
        upper = np.array([255, 255, 255])

        # The black region in the mask has the value of 0
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask = mask)

        # Gray scale
        gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Binary Mode
        ret,binary_gray = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)

        cv2.imwrite('Interface'+ str(img_index+1) + '.png', binary_gray)

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

            data_temp_x = []
            real_points = []

            print ('Frame#' + str(img_index+1) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
            data_temp_x.append(np.array(cY))
            data_temp_x.append(np.array(cX))
            dataX.append(data_temp_x)

            for j in dataX:
                cv2.circle(frame, (int(j[1]), int(j[0])), 1, (230, 0, 115), 1)
            cv2.namedWindow('See the trails', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('See the trails', frame)
            cv2.resizeWindow('See the trails', 900, 600)

            img_index += 1
            if img_index == 30:
                dataX = []
                img_index = 0
            cv2.waitKey(10)

    cap.realease()
    cv2.destroyAllWindows()
