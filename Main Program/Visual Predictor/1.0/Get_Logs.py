#!/usr/bin/python3

import datetime
import numpy as np

x = datetime.datetime.now()
x = (x.strftime("%c"))

def create_log(a,b,c):
    a = (np.array(a))
    b = (np.array(b))
    c = (np.array(c))
    with open('LOG - '+str(x)+'.log', 'w') as file:
            file.write('\n\nBUFFER\n\n' + str(a) + '\n\nREAL VALUES\n\n' + str(b) + '\n\nPREDICTED VALUES\n\n' + str(c))
            file.close()
