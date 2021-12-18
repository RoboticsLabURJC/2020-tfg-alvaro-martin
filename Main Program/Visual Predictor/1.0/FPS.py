#!/usr/bin/python3


def write_fps():
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)*5
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    # putting the FPS count on the frame
    frame = Preprocessing.resize(frame, 900, 600)
    frame = cv2.putText(frame, fps+"fps", (7, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
