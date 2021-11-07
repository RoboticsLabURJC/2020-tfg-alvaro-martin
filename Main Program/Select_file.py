#!/usr/bin/python3

import PySimpleGUI as sg

def Select_Video_File():

    layout2 = [[sg.Text('Source for File ', size=(15, 1)), sg.InputText(), sg.FileBrowse()],
             [sg.Submit(), sg.Cancel()]]

    window2 = sg.Window('Select Video File to get predictions', layout2)

    event2, values2 = window2.read()
    window2.close()
    file_path = values2[0]
    print(file_path)
    print(file_path.split('/')[-1])

    return file_path
