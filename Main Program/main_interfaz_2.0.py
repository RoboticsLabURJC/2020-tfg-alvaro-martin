#!/usr/bin/python3

import PySimpleGUI as sg
import Live_camera as LV
import Select_file as RV
import Get_trails
import Preprocessing
import Get_Logs
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
        #elif values["-LOGS-"] == True:
        #    print("LOGS")
        #elif event == 'Use Recorded File':
        elif values["-FILE-"]:
            window2.close()
            video = RV.Select_Video_File()
            if video and values["-LOGS-"] == True:
                dataX, GAP_data, FINAL = Get_trails.Extract_Frames(video)
                Get_Logs.create_log(dataX, GAP_data, FINAL)
            else:
                Get_trails.Extract_Frames(video)

        # Calculating the fps
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps)
        fps = str(fps)

        # putting the FPS count on the frame
        frame = Preprocessing.resize(frame, 900, 600)
        frame = cv2.putText(frame, fps+"fps", (7, 30), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window2["-IMAGE-"].update(data=imgbytes)
    window.close()
    window2.close()

main()
