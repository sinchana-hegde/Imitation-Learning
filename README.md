# Imitation-Learning
Drone learns to fly itself after being flown along a path once

Using datacollection.py, the frames seen by the drone is are recorded with their associated key strokes. This data is appended onto a file that contains the train_data.

This file is split into images and key strokes, which are passed in groups of "timesteps" for training. The architecture used is LRCN or Long Recurrent Convolutional Neural Network, which is a combination of CNN and LSTM.

The model is trained to predict the outcome based on "timesteps" number of frames seen. 

Upon runnig test_model.py, the drone flies along the path by itself.
