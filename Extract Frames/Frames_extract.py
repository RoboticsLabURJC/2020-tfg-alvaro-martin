# Extract all frames from a video using Python (OpenCV)
import cv2
import os
import numpy as np

folder_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Extract Frames/'
v = 0
cX = 0
previous_cX = 0

#for video in os.listdir(folder_path):
#    if video.endswith(".MP4") and not video.endswith("2501.MP4") and not video.endswith("2515.MP4") and not video.endswith("2491.MP4") and not video.endswith("2493.MP4") and not video.endswith("2490.MP4"):

        #video_name = video
video_name = 'MVI_2510.MP4'
video_path = folder_path + video_name

frames_path = folder_path + '/' + video_name + '_frames'
os.makedirs(frames_path,exist_ok=True)
cap = cv2.VideoCapture(video_path)
os.chdir(frames_path)
img_index = 0
v += 1
print('\nVideo #' + str(v) + '---------' + str(video_name))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imwrite('HSV' + str(img_index) + '.png', hsv)

    # Threshold of blue in HSV space
    lower_blue = np.array([120, 120, 120])
    upper_blue = np.array([255, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask = mask)


    #cv2.imwrite('video_frames_HSV' + str(img_index) + '.png', result)

    # Gray scale
    gray_image = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('GREY ' + str(img_index) + '.png', gray_image)

    #Binary Mode
    ret,binary_gray = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY_INV)
    #cv2.imwrite('video_frames_BIN ' + str(img_index) + '.png', binary_gray)

    kernel = np.ones((2,2), np.uint8)       # 3x3 matrix
    erosion_image = cv2.erode(binary_gray, kernel, iterations=1)
    cv2.imwrite('ERODE ' + str(img_index) + '.png', erosion_image)

    kernel = np.ones((10,10), np.uint8)     # 10x10 matrix
    dilation_image = cv2.dilate(erosion_image, kernel, iterations=1)
    cv2.imwrite('DILATE ' + str(img_index) + '.png', dilation_image)

    # calculate moments of binary image
    M = cv2.moments(dilation_image)

    # calculate x,y coordinate of center
    previous_cX = cX

    if int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0):
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv2.circle(gray_image, (cX, cY), 8, (255, 255, 255), 1)
        cv2.putText(gray_image, "centroid", (cX - 30, cY - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.circle(frame, (cX, cY), 8, (0, 0, 0), 1)
        cv2.putText(frame, "centroid", (cX - 30, cY - 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        if (cX > previous_cX):
            print(previous_cX, cX)
            print ('Frame#' + str(img_index) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
            cv2.imwrite('HVS+GREY+BIN(ERODE+DILATE) ' + str(img_index) + '.png', gray_image)
            cv2.imwrite('ORIGINAL ' + str(img_index) + '.png', frame)

        img_index += 1

        width = 120
        height = 80
        dim = (width, height)

        # resize image
        #resized = cv2.resize(black_image, dim, interpolation = cv2.INTER_AREA)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
