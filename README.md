# Imitation-Learning
Drone learns to fly itself after being flown along a path once

Using datacollection.py, the frames seen by the drone is are recorded with their associated key strokes. This data is appended onto a file that contains the train_data.

This file is split into images and key strokes, which are passed in groups of "timesteps" for training. The architecture used is LRCN or Long Recurrent Convolutional Neural Network, which is a combination of CNN and LSTM.

The model is trained to predict the outcome based on "timesteps" number of frames seen. 

Upon runnig test_model.py, the drone flies along the path by itself.

##Prerequisites##

1. Ubuntu 16.04

2. ROS Kinetic

```
http://wiki.ros.org/kinetic/Installation/Ubuntu
Visit this site and follow all the instructions (all the instructions)
```

3. OpenCV

4. TensorFlow GPU (For Training)

```
Follow instructions on
https://www.tensorflow.org/install/install_linux
```

5. TensorFlow CPU (For Testing)

```
Follow instructions on
https://www.tensorflow.org/install/install_linux
```


6. Python 2.7

7. CV Bridge (to interface between OpenCV and Python)

```
sudo apt-get install ros-kinetic-cv-bridge
```

8. Configuring Drone with Laptop 

```
sudo apt-get install build-essential python-rosdep python-catkin-tools


# Create and initialize the workspace

mkdir -p ~/bebop_ws/src && cd ~/bebop_ws
catkin init
git clone https://github.com/AutonomyLab/bebop_autonomy.git src/bebop_autonomy

# Update rosdep database and install dependencies (including parrot_arsdk)

rosdep update
rosdep install --from-paths src -i

# Build the workspace

catkin build

#Copy the teleop_key-master in folder PreRequisites to ~/bebop_ws/src

catkin build

```

9. Dependencies

```

Xlib
mss
numpy
pandas
future
pyautogui

sudo pip install xlib

#Similarly other dependencies

```

10. Folder Contents
```

There are three folders--> datacollection, training and testing
These folders contain the resp. scripts for each task.

```

11. Script Changes
```

Some Scipts have to be changed according to your laptop state and configuration:

1. mss_capture.py: 
    change the dimensions according to the postion & dimensions of the stream window.

2. automation.py:
    a. change the paths and names according to your machine & choice.
    b. Mouse commands in this scripts also need to be changed according to the positions of windows.


```

