#!/usr/bin/python3

import PySimpleGUI as sg

def Select_Video_File():

    layout2 = [[sg.Text('Source for File ', size=(15, 1)), sg.InputText(), sg.FileBrowse()],
             [sg.Submit(), sg.Cancel()]]

    window2 = sg.Window('Select Video File to get predictions', layout2, auto_size_text=True,
                       auto_size_buttons=True, resizable=True, grab_anywhere=False, border_depth=5,)

    event2, values2 = window2.read()
    window2.close()
    file_path = values2[0]

    return file_path
