---
title: "Previous weeks - Training the models"
excerpt: "With generated datashet and getting familiarized with the networks"

sidebar:
  nav: "docs"

classes: wide

categories:
- previous work

tags:
- logbook
- studying
- training

author: Álvaro Martín Menacho
pinned: false


---

## Training Networks

Now that we have the proper datashets created to train different models of Neural Networks with Deep Learning is time to check if the conclusions made by Nuria are the same with my models. I keeped the models which give her the least mean realative error after making the test.

For training the network I have been using the scripts main_train.py and net_train_config.yml located in [link](https://github.com/RoboticsLabURJC/2020-tfg-alvaro-martin/tree/main/Generator%20%26%20Train_Test/Network)

As you can see the code is very similar to the generator script, where you can chenge some basics like:
 - the complexity of the model (simple, complex, convLSTM, complex_convLSTM)
 - type of Net (Recurrent or Non recurrent), the activation function which decides, whether a neuron should be activated or not by calculating weighted sum and further adding bias with it
 - loss function (generate predictions, compare them with the actual values and then compute what is known as a loss, it has to be as low as possible)
 - batch data: taking data by groups, for large datashets
 - number of epochs
 - number of samples
 - patience: how many epochs are needed to determinate that the train have finished if there is no improve between epochs.



### Version
version: 1

### Complexity: simple, complex, convLSTM, complex_convLSTM
complexity: complex

### Root to save the model
root:  /Users/Martin/Desktop/TFG/Proyecto Github/2020-tfg-alvaro-martin/Generator & Train_Test/Models/
### root:  C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/Models/

### Type of the net to train (NoRec, Rec)
net_type: Rec

### Activation function
activation: relu
modeled_activation: tanh

### Loss function
raw_frame_loss: categorical_crossentropy
modeled_frame_loss: mean_squared_error

### Dropout options
dropout:
  flag: False
  percentage: 0.2

### Epochs
n_epochs: 5000

### Batch size
batch_size: 10

### Patience
patience: 15

### Data
data_dir: /Users/Martin/Desktop/_30/Frames_dataset/linear_var_5000_80_120/linear_30_[None]_
###data_dir: /Users/Martin/Desktop/Generator_10/Frames_dataset/parabolic_point_255_var_1_6000_80_120/parabolic_10_[None]_
###data_dir: /Users/Martin/Desktop/Generator_10/Frames_dataset/sinusoidal_point_255_fix_1000_80_120/sinusoidal_10_[None]_

### WINDOWS ······················································

### Data
###data_dir: C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/Datashet_10/Frames_dataset/linear_point_255_var_1_20000_80_120/linear_10_[None]_
###data_dir: C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/Datashet_10/Frames_dataset/parabolic_point_255_var_1_6000_80_120/parabolic_10_[None]_
###data_dir: C:/Users/optiva/Desktop/TFG/2020-tfg-alvaro-martin/Generator & Train_Test/Datashet_10/Frames_dataset/sinusoidal_point_255_fix_1000_80_120/sinusoidal_10_[None]_

batch_data: False ###True or False
data_model: modeled ###raw or modeled
gauss_pixel: False




So I started to understand how the code write in Python works and modified it for my own tests.
I created my firsts Neural Networks models with LSMT 1 layer (Rec-simple) and LSMT 4 layers(Rec-complex), which are the ones with better performance.



All the training models are located in [link](https://github.com/RoboticsLabURJC/2020-tfg-alvaro-martin/tree/main/Generator%20%26%20Train_Test/Models)

## LSTM1

{% include figure image_path="/assets/images/logbook/previous_weeks_2/NET1.png" alt="NET1" %}

{% include figure image_path="/assets/images/logbook/previous_weeks_2/Resultados_LSTM1.png" alt="Resultados_LSTM1" %}

## LSTM4

{% include figure image_path="/assets/images/logbook/previous_weeks_2/NET4.png" alt="NET4" %}

{% include figure image_path="/assets/images/logbook/previous_weeks_2/Resultados_LSTM4.png" alt="Resultados_LSTM4" %}


The extension of the files is .h5

10_False_tanh_mean_squared_error_10.h5
