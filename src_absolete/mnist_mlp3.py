'''
Trains a simple deep NN on the MNIST dataset and one extra class.


'''

from __future__ import print_function

import keras
from keras.preprocessing.image import img_to_array
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from PIL import Image
import glob
import numpy as np


#
# Configuration
#
batch_size = 128
num_classes = 11
epochs = 20


#
# Prepare custom class extention: 75:25
#
image_list = []
for filename in glob.glob('teach-data/A/*.png'):
    im=Image.open(filename)
    image_list.append(img_to_array(im))

x_train2 = np.array(image_list)
x_train2 = x_train2[:1350]
x_train2 = x_train2.reshape(1350, 784)

x_test2  = np.array(image_list)
x_test2  = x_test2[1350:1800]
x_test2  = x_test2.reshape(450, 784)

y_train2 = [10] * 1350
y_test2  = [10] * 450



#
# Prepare train and test sets
#
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)

x_train = np.concatenate((x_train, x_train2))
x_test  = np.concatenate((x_test,  x_test2 ))
y_train = np.concatenate((y_train, y_train2))
y_test  = np.concatenate((y_test,  y_test2 ))

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255


#
# Convert class vectors to binary class matrices
#
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)


#
# Construct the neural network
#
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='tensor-boards', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
					callbacks=[tbCallBack])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:',     score[0])
print('Test accuracy:', score[1])


#
# Real data recognition
#
img = Image.open('real-data/final_3.png')
img.load()
data = np.asarray(img, dtype="float32")
data /= 255
data = data.reshape(1, 784)

pred = model.predict_classes(data)
predVerbose = model.predict(data, batch_size=None, verbose=1)

print('Prediction: ', pred)
print('Prediction scores: \n', predVerbose)