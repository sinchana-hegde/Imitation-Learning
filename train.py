import keras
from keras.layers import Input ,Dense, Dropout, Activation, LSTM
from keras.layers import Lambda, Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
import keras.backend as K
from models import lrcn2 as lrcn
from models import alexnet2 as alex
from random import shuffle
import numpy as np
from keras.utils import to_categorical

timesteps=7;
number_of_samples=2500;
nb_samples=number_of_samples;
frame_row=32;
frame_col=32;
channels=3;

nb_epoch=100;
batch_size=timesteps;

FILE_I_END = 1

WIDTH = 100
HEIGHT = 100
LR = 1e-4
EPOCHS = 20

MODEL_NAME = 'lrcn-2.h5'

#data1= np.random.random((2500,timesteps,frame_row,frame_col,channels))
#label1=np.random.random((2500,timesteps,1))



file_name = 'data-13.npy'
train_data = np.load(file_name)

shuffle(train_data)

data = np.array([i[1] for i in train_data]).reshape(-1,timesteps,WIDTH,HEIGHT,3)
label = np.array([i[2] for i in train_data]).reshape(-1,timesteps,7)

#data2 - np.
X_train=data[0:2500,:]
y_train=label[0:2500]

#y_train = to_categorical(y_train)

X_test=data[2500:,:]
y_test=label[2500:]

#y_test = to_categorical(y_test)

model = lrcn(WIDTH, HEIGHT, 1, LR, output=7, model_name=MODEL_NAME)

model.summary()

#print(X_train.shape[1:])
##model=Sequential();                          

##model.add(TimeDistributed(Convolution2D(32, (3,3), strides =  (3,3), border_mode='same', input_shape=X_train.shape[1:])))
##model.add(TimeDistributed(Activation('relu')))
##model.add(TimeDistributed(Convolution2D(32, (3,3), strides = (3,3), activation = 'relu')))
##model.add(TimeDistributed(Activation('relu')))
##model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
##model.add(TimeDistributed(Dropout(0.25)))

##model.add(TimeDistributed(Flatten()))
##model.add(TimeDistributed(Dense(512)))



##model.add(TimeDistributed(Dense(35, name="first_dense" )))
#output dimension here is (None, 100, 35)                


##model.add(LSTM(output_dim=20, return_sequences=True))
#output dimension here is (None, 100, 20)

# time_distributed_merge_layer = Lambda(function=lambda x: K.mean(x, axis=1, keepdims=False))

# model.add(time_distributed_merge_layer)
#output dimension here is (None, 1, 20)


#model.add(Flatten())
##model.add(Dense(1, activation='sigmoid', input_shape=(None,20)))


##model.compile(loss='binary_crossentropy',
              #optimizer='adam',
              #metrics=['accuracy'])
model.summary()

model.fit(np.array(X_train), np.array(y_train), batch_size=batch_size, epochs =nb_epoch, validation_data=(np.array(X_test), np.array(y_test)))


print("Fit")

model.save(MODEL_NAME)
