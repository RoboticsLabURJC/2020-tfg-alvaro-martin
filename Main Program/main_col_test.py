#!/usr/bin/python3

import PySimpleGUI as sg
import Live_camera as LV
import Select_file as RV
import Get_trails
import Preprocessing
import cv2
import numpy as np
import time

def func(message):
    print(message)

def main():

    sg.theme("Dark Brown 4")


    # Define the window layout

    layout = [
                [sg.Text("Choose preferences before start", size=(60, 1), justification="center")],
                [sg.Radio("Use Live Recording", "Radio",True, key="-LIVE-"), sg.Radio("Use Recorded File", "Radio", key="-FILE-")],
                #[sg.Checkbox("Use Live Recording", default=True, key="-LIVE-")],
                [sg.Checkbox("Preprocessing", default=False, key="-PREPROCESING-"),],
                [sg.Checkbox("Prediction", default=False, key="-PREDICTIONS-"),],
                [sg.Checkbox("Log Records", default=False, key="-LOGS-"),],
                [sg.Text("Try this filters for Live camera", size=(60, 1), justification="center")],
                [sg.Radio("Binary", "Radio", size=(10, 1), key="-THRESH-"),
                        sg.Slider((0, 255),128,1,orientation="h",size=(40, 15),key="-THRESH SLIDER-",),],
                # HUE
                [sg.Radio("HUE", "Radio", size=(10, 1), key="-HUE-"),
                    sg.Slider((0, 225),0,1,orientation="h",size=(40, 15),key="-HUE SLIDER-",),],
                # CONTRAST
                [sg.Radio("Enhance", "Radio", size=(10, 1), key="-ENHANCE-"),
                    sg.Slider((1, 255),128,1,orientation="h",size=(40, 15),key="-ENHANCE SLIDER-",),],
                # BUTTON
                #[sg.Button("Use Recorded File"),
                [sg.Button("Exit")],
             ]
    # Create the window and show it without the plot

    layout2 = [[sg.Image(filename="", key="-IMAGE-")]]
    layout3 = [[sg.Image(filename="", key="-IMAGE-")]]

    window = sg.Window('Get predictions from video', layout)
    window2 = sg.Window('Live Video', layout2)
    window3 = sg.Window('Video File', layout3)

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

        #if values["-LIVE-"] == True:

        #if values["-LIVE-"] == False:
        #    break
        #    window2.close()
        if values["-THRESH-"]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
            frame = cv2.threshold(frame, values["-THRESH SLIDER-"], 255, cv2.THRESH_BINARY)[1]
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
        elif values["-PREPROCESING-"] == True:
            frame = Preprocessing.HSV_GRAY_BIN_ER_DIL(frame)
        elif values["-PREDICTIONS-"] == True:
            print("PREDICTIONS")
        elif values["-LOGS-"] == True:
            print("LOGS")
        #elif event == 'Use Recorded File':
        elif values["-FILE-"]:
            window2.close()
            video = RV.Select_Video_File()
            if video:
                cap2 = cv2.VideoCapture(video)
                while (cap.isOpened()):
                    event3, values3 = window3.read(timeout=20)
                    ret, frame = cap2.read()
                    if ret == False:
                        break
        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(new_frame_time-prev_frame_time)*5
        prev_frame_time = new_frame_time


        # converting the fps into integer
        fps = int(fps)


        # converting the fps to string so that we can display it on frame
        # by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        frame = Preprocessing.resize(frame, 900, 600)
        frame = cv2.putText(frame, fps+"fps", (7, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
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
