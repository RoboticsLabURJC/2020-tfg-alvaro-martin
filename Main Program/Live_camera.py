import numpy as np
import cv2

def Video_Live_Capture():

    cap = cv2.VideoCapture(0)

    while (True):
      ret, frame = cap.read()
      cv2.imshow('Visor', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.realease()
    cv2.destroyAllWindows()
