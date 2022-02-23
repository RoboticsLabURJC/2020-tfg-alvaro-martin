---
title: "Previous weeks - Finnally getting the correct results"
excerpt: "First valid models obteined"

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

## Getting the right models


After the productive meeting with Nuria I managed to obtain my own Neural Network models with LSTM1 and LSTM4 for the hole week I have been training models in order to use them in the next steps, use the prediction with the images captures by a camera.
For now, I only have been training with modeled samples dataset.

These are the results I obteined:

## Testing networks with MODELED samples

{% include figure image_path="/assets/images/logbook/previous_weeks_4/Resultados.png" alt="Resultados" %}

I continue the training of different networks with the generated data.
Also I add some grapchs to have a visual information about how the test was going on.

## Train vs Validation

{% include figure image_path="/assets/images/logbook/previous_weeks_4/Val_loss.png" alt="Val_loss" %}

## LSTM 1 with 6000 samples

{% include figure image_path="/assets/images/logbook/previous_weeks_4/LSTM1_6000.png" alt="LSTM1_6000" %}

## LSTM 4 with 20000 samples

{% include figure image_path="/assets/images/logbook/previous_weeks_4/LSTM1_6000.png" alt="LSTM4_20000" %}

For the neext week I will be training with raw frames
