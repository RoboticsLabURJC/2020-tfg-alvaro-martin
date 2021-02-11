# Extract all frames from a video using Python (OpenCV)
import cv2
import os

folder_path = '/Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Extract Frames/'
video_name = 'MVI_9993.MOV'
video_path = folder_path + video_name

frames_path = folder_path + '/frames' + '_' + video_name + '/'
os.mkdir(frames_path)

cap = cv2.VideoCapture(video_path)
os.chdir(frames_path)
img_index = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    gray_effect = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        break
    cv2.imwrite('video_frames' + str(img_index) + '.png', gray_effect)
    img_index += 1

cap.release()
cv2.destroyAllWindows()
