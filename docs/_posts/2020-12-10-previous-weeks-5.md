---
title: "Previous weeks - Testing RAW models"
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

## Testing networks with RAW samples

This week I was testing the training models with raw samples instead of modeled.

These are the results I obteined:

{% include figure image_path="/assets/images/logbook/previous_weeks_5/Resultados.png" alt="Resultados" %}

## Train vs Validation

{% include figure image_path="/assets/images/logbook/previous_weeks_5/Val_loss.png" alt="Val_loss" %}

## convLSTM 1 with 1000 samples Linear 1DOF

{% include figure image_path="/assets/images/logbook/previous_weeks_5/convLSTM1_1000_parabolic_1DOF.png" alt="convLSTM1_1000_parabolic_1DOF" %}

## convLSTM 1 with 1000 samples Linear 2DOF

{% include figure image_path="/assets/images/logbook/previous_weeks_5/convLSTM1_1000_sinusoidal_1DOF.png" alt="convLSTM1_1000_sinusoidal_1DOF" %}


As you can see it was difficult to obtain a good result with raw images and the cappacities of my computer are not the best for this type of works. So Jose María recommended me to train and test the results with even more samples and use a GPU provided by URJC university to get the .h5 files done and use them for the final objective, work with real frames captured by a camera.
