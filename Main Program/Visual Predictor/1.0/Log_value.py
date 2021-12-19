#!/usr/bin/python3

import logging
import datetime
import numpy as np

x = datetime.datetime.now()
x = (x.strftime("%c"))

def create_log():
    logging.basicConfig(
        format = '%(asctime)-5s %(levelname)-8s %(message)s',
        level  = logging.DEBUG,
        filename='EVERY VALUE - '+str(x)+'.log',
        filemode='a',
    )
    logging.info('Frame extraction extarting...')
    logging.info('Getting buffer values')

def add_centroid(frame, img_index):
    frame = (np.array(frame))

    logging.info('Frame nº '+ str(img_index) + ':' + str(frame))

def add_predicted(real, predicted, img_index):
    real = (np.array(real))
    predicted = (np.array(predicted))

    logging.debug('Frame nº '+ str(img_index) + ';       Real value to predict... ' + str(real))
    logging.debug('Frame nº '+ str(img_index) + ';       Predicted value... ' + str(predicted))

def bad_prediction_x(real, predicted):
    real = (np.array(real))
    predicted = (np.array(predicted))

    logging.warning('----- BAD PREDICTION. More than 5 pixels of differential in X axis  ----- Real value... ' + str(real) + ' vs Predicted value... ' + str(predicted))

def bad_prediction_y(real, predicted):
    real = (np.array(real))
    predicted = (np.array(predicted))

    logging.warning('----- BAD PREDICTION. More than 5 pixels of differential in Y axis  ----- Real value... ' + str(real) + ' vs Predicted value... ' + str(predicted))

def good_prediction(real, predicted):
    real = (np.array(real))
    predicted = (np.array(predicted))

    logging.debug('----- GOOD PREDICTION  -----')

def end_log():
    logging.info('Video Prediction ended...')
    logging.shutdown()
