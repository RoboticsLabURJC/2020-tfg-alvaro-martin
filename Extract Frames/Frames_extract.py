# Extract all frames from a video using Python (OpenCV)
import cv2
import os
import numpy as np

folder_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Extract Frames/'
video_name = 'MVI_2481.MP4'
video_path = folder_path + video_name

frames_path = folder_path + '/frames' + '_' + video_name
is_path = os.path.isdir(frames_path)
print(is_path)
if is_path == False:
    os.mkdir(frames_path)
    cap = cv2.VideoCapture(video_path)
    os.chdir(frames_path)
    img_index = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,binary = cv2.threshold(gray_image,127,255,0)
        width = 120
        height = 80
        dim = (width, height)

        # resize image
        resized = cv2.resize(binary, dim, interpolation = cv2.INTER_AREA)

        # calculate moments of binary image
        M = cv2.moments(resized)
        # calculate x,y coordinate of center

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # put text and highlight the center
        black_image = np.zeros((80,120,3), np.uint8)

        #cv2.circle(black_image, (cX, cY), 1, (255, 255, 255), 1)
        print (cX,cY)
        if (cY <= 120) and (cX <= 80):
            black_image[(cY, cX)] = 255
            if img_index >= 135 and img_index <= 180:
            cv2.imwrite('video_frames' + str(img_index) + '.png', black_image )

        img_index += 1

        #if ((img_index % 10) == 0):

    cap.release()
    cv2.destroyAllWindows()

else:
    print('The folder already exist')
