---
title: "Previous weeks - Testing MODELED samples"
excerpt: "Getting results with max relative mean error"

sidebar:
  nav: "docs"

classes: wide

categories:
- previous work

tags:
- logbook
- studying
- testing

author: Álvaro Martín Menacho
pinned: false


---

## Testing networks

So now we have the generated dataset and the network model trained with that data, we need to test it and see if our model is valid.
For I have been using the scripts main_test.py and net_test_config.yml located in [link](https://github.com/RoboticsLabURJC/2020-tfg-alvaro-martin/tree/main/Generator%20%26%20Train_Test/Network), very similar to train scripts.


Basically, what the test script do is the finction:

to_test_net.test(testX, testY, gap, data_type, dim)

where:

textX -> Array of 20 values (x,y)
textY -> Array of 1 value (x,y)
gap -> Gap between the last value of testX and testY, meaning how many frames after is the network going to predict the pixel values.
data_type -> raw or modeled
dim -> dimensions of the frame, the default I have been using is 120x80


First I started with simple modeled images around 2000-4000 samples (80% training, 10% test, 10% validation) with DOF 1 and 2 (linear and parabolic)


The results differ a lot from the ones Nuria achieve.
I was getting around 3% - 6% mean squared error for LSTM1 with 2000 samples and her results were around 0,1% - 0,3%
The problem was that I was using only a few samples comparing with her (2000 vs 20000, 10000 vs 100000) and also some configuration wasn't working properly, like batch size, dropout and I was suffering overfitting in some cases.

For next week I have to take the appropriate changes and get the right model to predict my frames in real life.
I planned a meeting with her to get all the info neccesary.
