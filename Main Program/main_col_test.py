#!/usr/bin/python3

import PySimpleGUI as sg
import Live_camera as LV
import Select_file as RV
import Get_trails
import cv2
import numpy as np

def main():

    sg.theme("LightGreen")


    # Define the window layout

    l_col = sg.Column([[sg.Text("OpenCV Demo", size=(60, 1), justification="center")]])
            #[sg.Image(filename="", key="-IMAGE-")]],


    r_col = sg.Column([sg.Radio("None", "Radio", True, size=(10, 1))],
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
            element_justification="right",
            vertical_alignment="bottom",
            expand_x=True,
            )

    layout = [[sg.Frame(layout=[[l_col, r_col],],title="Cool subpanel",)]]
    # Create the window and show it without the plot

    window = sg.Window('Get predictions from video', layout)

    cap = cv2.VideoCapture(0)

    while True:

        event, values = window.read(timeout=20)
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
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
    window.close()

main()


'''
l_col = sg.Column(
    [
        [
            sg.Text("First parameter", size=(15, 1)),
            sg.InputText(default_text="2", size=(3, 1)),
        ],
        [
            sg.Text("Second parameter", size=(15, 1)),
            sg.InputText(default_text="8", size=(3, 1)),
        ],
    ]
)
r_col = sg.Column(
    [[sg.Submit("A nice button", size=(23, 1))]],
    element_justification="right",
    vertical_alignment="bottom",
    expand_x=True,
)

layout = [
    [
        sg.Text("Target folder", size=(9, 1)),
        sg.InputText(default_text="Choose a folder...", size=(59, 1)),
        sg.FolderBrowse(),
    ],
    [
        sg.Frame(
            layout=[
                [l_col, r_col],
                [sg.ProgressBar(1, orientation="h", size=(50, 20))],
            ],
            title="Cool subpanel",
            relief=sg.RELIEF_GROOVE,
        )
    ],
]

    while True:             # Event Loop
        event, values = window.Read()
        if event in (None, 'Exit'):
            break

    window.Close()

main()

'''
