#!/usr/bin/python3

import PySimpleGUI as sg
import Live_camera as LV
import Select_file as RV
import Get_trails
import cv2
import numpy as np
import time

def main():

    sg.theme("LightGreen")


    # Define the window layout

    layout = [
                [sg.Text("Get predictions from video", size=(60, 1), justification="center")],
                [sg.Button('Use Live Recording'), sg.Button('Use Recorded File')],
                [sg.Radio("None", "Radio", True, size=(10, 1))],
                    [sg.Radio("threshold", "Radio", size=(10, 1), key="-THRESH-"),
                        sg.Slider((0, 255),128,1,orientation="h",size=(40, 15),key="-THRESH SLIDER-",),],
                # CONTROL
                [sg.Radio("canny", "Radio", size=(10, 1), key="-CANNY-"),
                    sg.Slider((0, 255),128,1,orientation="h",size=(20, 15),key="-CANNY SLIDER A-",),
                        sg.Slider((0, 255),128,1,orientation="h",size=(20, 15),key="-CANNY SLIDER B-",),],
                #BLUR
                [sg.Radio("blur", "Radio", size=(10, 1), key="-BLUR-"),
                    sg.Slider((1, 11),1,1,orientation="h",size=(40, 15),key="-BLUR SLIDER-",),],
                # HUE
                [sg.Radio("hue", "Radio", size=(10, 1), key="-HUE-"),
                    sg.Slider((0, 225),0,1,orientation="h",size=(40, 15),key="-HUE SLIDER-",),],
                # CONTRAST
                [sg.Radio("enhance", "Radio", size=(10, 1), key="-ENHANCE-"),
                    sg.Slider((1, 255),128,1,orientation="h",size=(40, 15),key="-ENHANCE SLIDER-",),],
                # BUTTON
                [sg.Button("Exit", size=(10, 1))],
             ]
    # Create the window and show it without the plot

    layout2 = [[sg.Image(filename="", key="-IMAGE-")]]

    window = sg.Window('Get predictions from video', layout)
    window2 = sg.Window('Video', layout2)

    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    cap = cv2.VideoCapture(0)

    while True:

        event, values = window.read(timeout=20)
        event2, values2 = window2.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        ret, frame = cap.read()

        if values["-THRESH-"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
            frame = cv2.threshold(frame, values["-THRESH SLIDER-"], 255, cv2.THRESH_BINARY)[1]
        elif values["-CANNY-"]:
            frame = cv2.Canny(frame, values["-CANNY SLIDER A-"], values["-CANNY SLIDER B-"])
        elif values["-BLUR-"]:
            frame = cv2.GaussianBlur(frame, (21, 21), values["-BLUR SLIDER-"])
        elif values["-HUE-"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[:, :, 0] += int(values["-HUE SLIDER-"])
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        elif values["-ENHANCE-"]:
            enh_val = values["-ENHANCE SLIDER-"] / 40
            clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)*10
        print(fps)
        print(prev_frame_time)
        print(new_frame_time)
        prev_frame_time = new_frame_time


        # converting the fps into integer
        fps = int(fps)


        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        frame = cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window2["-IMAGE-"].update(data=imgbytes)
    window.close()
    window2.close()

main()


'''
layout = [[sg.Button('Use Live Recording'), sg.Button('Use Recorded File'), sg.Exit()]]

# image_filename ='play.png', image_size =(50, 50))

window = sg.Window('Get predictions from video', layout, auto_size_text=True,
                   auto_size_buttons=True, resizable=True, grab_anywhere=True, border_depth=5)


while True:             # Event Loop
    event, values = window.Read()
    if event in (None, 'Exit'):
        break
    if event == 'Use Live Recording':
        func('Live Video Recording Selected')
        LV.Video_Live_Capture()
    elif event == 'Use Recorded File':
        func('Use Recorded File')
        video = RV.Select_Video_File()
        if video:
            Get_trails.Extract_Frames(video)
window.Close()



'''
