# Extract all frames from a video using Python (OpenCV)
import cv2

video_path = 'D:\\PROGRAMMING\\3-Python\\My_Virtual_Envs\\opencv_extract_frames\\video_ok.mp4'
cap = cv2.VideoCapture(video_path)

img_index = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    gray_effect = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        break
    cv2.imwrite('myphotos' + str(img_index) + '.png', gray_effect)
    img_index += 1

cap.release()
cv2.destroyAllWindows()
