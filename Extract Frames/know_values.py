import cv2
import os
import numpy as np

folder_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Extract Frames/'
video_name = 'MVI_2510.MP4'
video_path = folder_path + video_name
cX = 0
previous_cX = 0

frames_path = folder_path + '/' + video_name + '_frames'
os.makedirs(frames_path,exist_ok=True)
cap = cv2.VideoCapture(video_path)
os.chdir(frames_path)
img_index = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.bitwise_not(hsv) # Inverted filter color


    # Threshold in HSV space
    lower = np.array([0, 145, 0]) # Green Background
    #lower = np.array([0, 0, 160]) # Black background
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask = mask)

    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    #Binary Mode
    ret,binary_gray = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)       # 3x3 matrix
    erosion_image = cv2.erode(binary_gray, kernel, iterations=1)

    kernel = np.ones((5,5), np.uint8)     # 10x10 matrix
    dilation_image = cv2.dilate(erosion_image, kernel, iterations=1)

    M = cv2.moments(dilation_image)

    previous_cX = cX

    if int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.circle(dilation_image, (cX, cY), 6, (0, 0, 0), 1)
        cv2.putText(dilation_image, "here", (cX - 10, cY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.circle(frame, (cX, cY), 6, (0, 0, 0), 1)
        cv2.putText(frame, "here", (cX - 10, cY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if (cX > previous_cX):
            print(previous_cX, cX)
            print ('Frame#' + str(img_index) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
            cv2.imwrite('HSV' + str(img_index) + '.png', hsv)
            cv2.imwrite('MASK ' + str(img_index) + '.png', mask)
            cv2.imwrite('GREY ' + str(img_index) + '.png', gray_image)
            cv2.imwrite('BINARY ' + str(img_index) + '.png', binary_gray)
            cv2.imwrite('ERODE ' + str(img_index) + '.png', erosion_image)

            cv2.imwrite('HVS+GREY+BIN(ERODE+DILATE) ' + str(img_index) + '.png', dilation_image)
            cv2.imwrite('ORIGINAL ' + str(img_index) + '.png', frame)

        cv2.waitKey(0)
    img_index += 1

cap.release()
cv2.destroyAllWindows()
