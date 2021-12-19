#!/usr/bin/python3

import logging
import datetime
import numpy as np

x = datetime.datetime.now()
x = (x.strftime("%c"))

def create_log(a,b,c):
    a = (np.array(a))
    b = (np.array(b))
    c = (np.array(c))
    # Creación del logger que muestra la información únicamente por consola.
    logging.basicConfig(
        format = '%(asctime)-5s %(levelname)-8s %(message)s',
        level  = logging.DEBUG,
        filename='LOG - '+str(x)+'.log',
        filemode='w',
    )
    logging.info('Frame extraction extarting')
    logging.info('BUFFER VALUES')
    n = 0
    for i in a:
        logging.info('Frame nº '+ str(n) + ':' + str(i))
        n +=1
    n = 0
    logging.info('REAL VALUES')
    for i in b:
        logging.info('Real nº '+ str(n) + ':' + str(i))
        n +=1
    logging.info('PREDICTED VALUES')
    n = 0
    for i in c:
        logging.info('Precited nº '+ str(n) + ':' + str(i))
        n +=1
    logging.info('Ending prediction')
    logging.shutdown()
