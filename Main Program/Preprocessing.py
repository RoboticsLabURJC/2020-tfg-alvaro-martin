#!/usr/bin/python3

import cv2
import numpy as np

''' Resize image '''
def resize(img, width, height):

    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized

def HSV_GRAY_BIN_ER_DIL(frame):

  ''' FRAME RESIZED '''
  #frame = resize(frame, 120, 80)

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

  return dilation_image
