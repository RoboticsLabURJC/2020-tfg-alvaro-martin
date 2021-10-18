import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (True):
  ret, frame = cap.read()
  cv2.imshow('Visor', frame)

  if cv2.watiKey(1) & 0xFF == ord('q'):
    break

cap.realease()
cv2.destroyAllWindows()
