#!/usr/bin/python3

import Live_camera as LV

def MENU():

    correcto=False
    num=0
    while(not correcto):
        try:
            num = int(input("Introduce un número para continuar: "))
            correcto=True
        except ValueError:
            print('Error, introduce un número entero')

    return num

salir = False
opcion = 0

while not salir:

    print ("1. Live Camera")
    print ("2. Choose Recorded Video")
    print ("3. Opcion 3")
    print ("4. Salir")

    print ("Elige una opcion")

    opcion = pedirNumeroEntero()

    if opcion == 1:
        LV.Video_Live_Capture()
    elif opcion == 2:
        print ("Choose Recorded Video")
    elif opcion == 3:
        print("Opcion 3")
    elif opcion == 4:
        salir = True
    else:
        print ("Introduce un numero entre 1 y 3")

print ("Fin")
