#!/usr/bin/python3

import logging
import datetime
import numpy as np

x = datetime.datetime.now()
x = (x.strftime("%c"))

def add_buffer(a):
    a = (np.array(a))
    # Creación del logger que muestra la información únicamente por consola.
    logging.basicConfig(
        format = '%(asctime)-5s %(levelname)-8s %(message)s',
        level  = logging.DEBUG,
        filename='EVERY VALUE - '+str(x)+'.log',
        filemode='a',
    )
    logging.debug('Frame extraction extarting...')
    logging.info('Getting buffer values')
    n = 0
    for i in a:
        logging.info('Frame nº '+ str(n) + ':' + str(i))
        n +=1
    logging.shutdown()

def add_real(b):
    b = (np.array(b))
    print(str(b))
    # Creación del logger que muestra la información únicamente por consola.
    logging.basicConfig(
        format = '%(asctime)-5s %(levelname)-8s %(message)s',
        level  = logging.DEBUG,
        filename='EVERY VALUE - '+str(x)+'.log',
        filemode='a',
    )
    logging.debug('Real value to predict...')
    logging.info('Real nº: ' + str(b))
    logging.shutdown()


def add_predicted(c):
    c = (np.array(c))
    # Creación del logger que muestra la información únicamente por consola.
    logging.basicConfig(
        format = '%(asctime)-5s %(levelname)-8s %(message)s',
        level  = logging.DEBUG,
        filename='EVERY VALUE - '+str(x)+'.log',
        filemode='a',
    )
    logging.debug('Predicted value')
    logging.info('Predicted nº: ' + str(c))
    logging.shutdown()
