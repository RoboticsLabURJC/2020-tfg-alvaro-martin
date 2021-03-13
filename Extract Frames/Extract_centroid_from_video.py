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
def resize(width, height):

    dim = (width, height)
    resized = cv2.resize(black_image, dim, interpolation = cv2.INTER_AREA)

    return resized


def Extract_Frames():

    v = 0
    cX = 0
    previous_cX = 0

    # Create new blank image
    w, h = 1280, 720
    black = (0, 0, 0)
    black_img = create_blank(w, h, rgb_color=black)

    # List all the videos
    for video in os.listdir(folder_path):
        if video.endswith(".MP4"):
            video_name = video
            video_path = folder_path + video
            img_index = 0
            v += 1

            frames_path = folder_path + '/' + video_name + '_frames'
            os.makedirs(frames_path,exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            os.chdir(frames_path)

            print('\nVideo #' + str(v) + '---------' + str(video_name))

            # List all the frames
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break

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

                    cv2.circle(black_img, (cX, cY), 5, (246, 209, 81), -1)
                    cv2.imwrite('Real_Trails.png', black_img)
                    cv2.circle(dilation_image, (cX, cY), 6, (0, 0, 0), 1)
                    cv2.putText(dilation_image, "here", (cX - 10, cY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.circle(frame, (cX, cY), 6, (0, 0, 0), 1)
                    cv2.putText(frame, "here", (cX - 10, cY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    if (cX > previous_cX):
                        print(previous_cX, cX)
                        print ('Frame#' + str(img_index) + ' centroid ----- ' + str(cX) + ' ' + str(cY))
                        cv2.imwrite('HVS+GREY+BIN(ERODE+DILATE) ' + str(img_index) + '.png', dilation_image)
                        cv2.imwrite('ORIGINAL ' + str(img_index) + '.png', frame)

                    img_index += 1
                    cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    folder_path = '/Users/Martin/Desktop/Nuevas tomas/'
    Extract_Frames()
    print("\n----- DONE. ALL IMAGES PROCESSED -----\n")
