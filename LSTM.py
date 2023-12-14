
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns  # for nicer plots
sns.set(style="darkgrid")  # default style
import plotly.graph_objs as plotly  # for interactive plots

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM
from keras.layers import TimeDistributed
import os
import pickle
import neuralNetwork as nnet


AccuracyList = []

def build_LSTM(kernel_size=(5,5),strides=(1, 1),pool_size=(2, 2),learning_rate=0.001,opt='Adam', cnn=False, hidden_layer_sizes=[]):
  
    model = tf.keras.Sequential()
    # add first convolution layer to the model
    input_shape=( 128,128, 1)
 
    if cnn:
    # add first convolution layer to the model
        for i in range(len(hidden_layer_sizes)):
            model.add(tf.keras.layers.Conv2D(filters=hidden_layer_sizes[i],kernel_size=kernel_size,strides=strides,
                padding='same',data_format='channels_last',name=f'conv_{i}',activation='relu'))
        # add a max pooling layer with pool size and strides 
            model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size,name=f'pool_{i}'))

        
    # add a fully connected layer (need to flatten the output of the previous layers first)
        model.add(TimeDistributed(tf.keras.layers.Flatten()) )
        model.add(TimeDistributed(tf.keras.layers.Dense(units=1024,name='fc_1', activation='relu')))

    # add dropout layer
        model.add(TimeDistributed(tf.keras.layers.Dropout(rate=0.5)))

    model.add(LSTM(1024, return_sequences=False, input_shape=input_shape))

    model.add(tf.keras.layers.Dense(units=6, activation='softmax'))

    # build model and print summary
    tf.random.set_seed(1)
    model.build(input_shape=(None, 128, 128, 1))
    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy']) 
    return model

def train_and_evaluate(X_train, X_test, Y_train, Y_test, file_name='', num_epochs = 5, batch_size=32, params={}):
    
    model= None
    history = None

    if os.path.exists(file_name + '.keras'):
        model = tf.keras.models.load_model(file_name + '.keras')
        with open(file_name + '.pkl', 'rb') as file:
            history = pickle.load(file)
        print(f"The file {file_name} already exists.")

    else:
        model = build_LSTM(**params)
        X_train = tf.expand_dims(X_train, axis=-1)  # Assuming grayscale, add channel dimension
        X_test = tf.expand_dims(X_test, axis=-1)
        print('Training...')
        history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size= batch_size, validation_data=(X_test, Y_test))
    
        model.save(file_name + '.keras')
        # Save the training history separately
        with open(file_name + '.pkl', 'wb') as file:
            pickle.dump(history, file)
    
    # Retrieve the training metrics (after each train epoch) and the final test
    # accuracy.

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    test_accuracy = model.evaluate(x=X_test, y=Y_test, verbose=0, return_dict=True)['accuracy']
    title = 'lstm'
    for key in params.keys():
        title += '-' + key
    accuracy = nnet.AccuracyClass(train_accuracy, val_accuracy, num_epochs, title, model, test_accuracy )
    AccuracyList.append(accuracy)

    return test_accuracy