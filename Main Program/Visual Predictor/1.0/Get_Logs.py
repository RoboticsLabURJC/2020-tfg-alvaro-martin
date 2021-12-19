#!/usr/bin/python3

import datetime
import logging
import numpy as np

x = datetime.datetime.now()
x = (x.strftime("%c"))

def create_log(a,b,c):
    a = (np.array(a))
    b = (np.array(b))
    c = (np.array(c))

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')
    logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

    with open('LOG - '+str(x)+'.log', 'w') as file:
            file.write('\n\nBUFFER\n\n' + str(a) + '\n\nREAL VALUES\n\n' + str(b) + '\n\nPREDICTED VALUES\n\n' + str(c))
            file.close()
