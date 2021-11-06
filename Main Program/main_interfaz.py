#!/usr/bin/python3

import PySimpleGUI as sg
import Live_camera as LV
import Select_file as RV

def func(message):
    print(message)

layout = [[sg.Button('Use Live Recording'), sg.Button('Use Recorded File'), sg.Exit()] ]

window = sg.Window('Get predictions from video').Layout(layout)

while True:             # Event Loop
    event, values = window.Read()
    if event in (None, 'Exit'):
        break
    if event == 'Use Live Recording':
        func('Live Video Recording Selected')
        LV.Video_Live_Capture()
    elif event == 'Use Recorded File':
        func('Use Recorded File')
        RV.Select_Video_File()
window.Close()
