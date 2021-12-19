#!/usr/bin/python3

import Preprocessing
import cv2
import time

def write_fps(frame, fps):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = int(fps)
    fps = str(fps)

    # putting the FPS count on the frame
    frame = Preprocessing.resize(frame, 900, 600)
    frame = cv2.putText(frame, fps+"fps", (7, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    return frame
