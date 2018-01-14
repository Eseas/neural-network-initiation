'''
Trains a simple deep NN on the MNIST dataset and one extra class.


'''

from __future__ import print_function

import keras
from keras.preprocessing.image import img_to_array
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from PIL import Image
import glob
import numpy as np

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras import backend as K

#
# Configuration
#
MODEL_NAME = 'mnist_mlp1'
batch_size = 128
num_classes = 11
epochs = 1


def main():
    #
    # Prepare custom class extention: 75:25
    #
    image_list = []
    for filename in glob.glob('teach-data/A/*.png'):
        # assert isinstance(filename, str) # doesn't seem to do anything
        im = Image.open(filename)
        image_list.append(img_to_array(im))
        dim_test = img_to_array(im)
        print(dim_test.shape)

    x_train2 = np.array(image_list)
    x_train2 = x_train2[:1350]
    x_train2 = x_train2.reshape(1350, 28, 28)
    print(x_train2.shape)

    x_test2 = np.array(image_list)
    x_test2 = x_test2[1350:1800]
    x_test2 = x_test2.reshape(450, 28, 28)
    print(x_test2.shape)

    y_train2 = [10] * 1350
    y_test2 = [10] * 450



    #
    # Prepare train and test sets
    #
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # x - uint8 arrays of grayscale image data with shape(num_samples, 28, 28)
    x_train = x_train.astype('float32')
    x_train = x_train.reshape(60000, 28, 28)  # SKIP CAUSE IT'S ALREADY 28x28?????
    x_test = x_test.reshape(10000, 28, 28)  # SKIP CAUSE IT'S ALREADY 28x28?????
    print(x_train.shape)

    x_train = np.concatenate((x_train, x_train2))
    x_test = np.concatenate((x_test,  x_test2 ))
    y_train = np.concatenate((y_train, y_train2))
    y_test = np.concatenate((y_test,  y_test2 ))

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
    model.add(Dense(512, activation='relu', input_shape=( 28, 28)))
    model.add(Flatten()) # change shape
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
    # img = Image.open('real-data/final_3.png')
    # img.load()
    # data = np.asarray(img, dtype="float32")
    # data /= 255
    # data = data.reshape(1, 784)
    #
    # pred = model.predict_classes(data)
    # predVerbose = model.predict(data, batch_size=None, verbose=1)
    #
    # print('Prediction: ', pred)
    # print('Prediction scores: \n', predVerbose)

    # Export weights

    export_model(tf.train.Saver(), model, ["dense_1_input"], "dense_3/Softmax")


# This function is taken from
# https://github.com/llSourcell/A_Guide_to_Running_Tensorflow_Models_on_Android/blob/master/tensorflow_model/mnist_convnet_keras.py

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out2', MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out2/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out2/' + MODEL_NAME + '_graph.pbtxt', None, False, 'out2/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all", "save/Const:0", 'out2/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out2/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out2/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


if __name__ == '__main__':
    main()