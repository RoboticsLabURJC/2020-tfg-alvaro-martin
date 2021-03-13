# Extract all frames from a video using Python (OpenCV)
import cv2
import os
import numpy as np

folder_path = 'C:/Users/optiva/Desktop/Nuevas tomas/'
v = 0
cX = 0
previous_cX = 0

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# Create new blank 300x300 black image
width1, height1 = 500, 500

black = (0, 0, 0)
black_img = create_blank(width1, height1, rgb_color=black)

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

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.bitwise_not(hsv) # Inverted filter color
            #cv2.imwrite('HSV' + str(img_index) + '.png', hsv)

            # Threshold in HSV space
            lower = np.array([225, 0, 0]) # Green Background
            #lower = np.array([0, 0, 160]) # Black background
            upper = np.array([255, 255, 255])

            # The black region in the mask has the value of 0
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask = mask)
            #cv2.imwrite('video_frames_MASK' + str(img_index) + '.png', result)

            # Gray scale
            gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite('GREY ' + str(img_index) + '.png', gray_image)

            # Binary Mode
            ret,binary_gray = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)
            #cv2.imwrite('video_frames_BIN ' + str(img_index) + '.png', binary_gray)

            kernel = np.ones((3,3), np.uint8)       # 3x3 matrix
            erosion_image = cv2.erode(binary_gray, kernel, iterations=1)
            #cv2.imwrite('ERODE ' + str(img_index) + '.png', erosion_image)

            kernel = np.ones((5,5), np.uint8)     # 10x10 matrix
            dilation_image = cv2.dilate(erosion_image, kernel, iterations=1)
            #cv2.imwrite('DILATE ' + str(img_index) + '.png', dilation_image)

            # calculate moments of binary image
            M = cv2.moments(dilation_image)
            previous_cX = cX

            #mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

            if int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0) or int((M["m10"]) != 0):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.circle(black_img, (cX, cY), 3, (150, 0, 0), -1)
                cv2.imwrite('Estela_azul.png', black_img)

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


                # resize image
                #width = 120
                #height = 80
                #dim = (width, height)
                #resized = cv2.resize(black_image, dim, interpolation = cv2.INTER_AREA)
                cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
