
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns  # for nicer plots
sns.set(style="darkgrid")  # default style
import plotly.graph_objs as plotly  # for interactive plots

import tensorflow as tf
from tensorflow import keras
import os
import pickle


AccuracyList = []

def build_CNN(kernel_size=(5,5),strides=(1, 1),pool_size=(2, 2),learning_rate=0.001,opt='Adam'):
  
    CNNmodel = tf.keras.Sequential()

    # add first convolution layer to the model
    CNNmodel.add(tf.keras.layers.Conv2D(filters=32,kernel_size=kernel_size,strides=strides,
        padding='same',data_format='channels_last',name='conv_1',activation='relu'))

    # add a max pooling layer with pool size and strides 
    CNNmodel.add(tf.keras.layers.MaxPool2D(pool_size=pool_size,name='pool_1'))

    # add second convolutional layer
    CNNmodel.add(tf.keras.layers.Conv2D(filters=64,kernel_size=kernel_size,strides=strides,
        padding='same',name='conv_2',activation='relu'))

    # add second max pooling layer with pool size and strides of 2
    CNNmodel.add(tf.keras.layers.MaxPool2D(pool_size=pool_size, name='pool_2'))


    # add a fully connected layer (need to flatten the output of the previous layers first)
    CNNmodel.add(tf.keras.layers.Flatten()) 
    CNNmodel.add(tf.keras.layers.Dense(units=1024,name='fc_1', activation='relu'))

    # add dropout layer
    CNNmodel.add(tf.keras.layers.Dropout(rate=0.5))

    # add the last fully connected layer
    #CNNmodel.add(tf.keras.layers.Dense(units=1,name='fc_2',activation=None))
    CNNmodel.add(tf.keras.layers.Dense(6, activation='softmax'))

    # build model and print summary
    tf.random.set_seed(1)
    CNNmodel.build(input_shape=(None, 128, 128, 1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    CNNmodel.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy']) 
    
    return CNNmodel

class AccuracyClass:
  def __init__(self, train_accuracy , val_accuracy, num_epochs, title, model, test_accuracy):
    self.train_accuracy = train_accuracy
    self.val_accuracy = val_accuracy
    self.num_epochs = num_epochs
    self.title = title
    self.model = model
    self.test_accuracy = test_accuracy
  
  def plot(self):
    plt.plot(self.train_accuracy, label='train_accuracy')
    plt.plot(self.val_accuracy, label='validation accuracy')
    plt.xticks(range(self.num_epochs))
    plt.xlabel('Train epochs')
    plt.title(self.title)
    plt.legend()
    plt.show()

def train_and_evaluate(X_train, X_test, Y_train, Y_test, file_name='', num_epochs = 5, batch_size=32, params={}, train_datagen=None):
    
    model= None
    history = None

    if os.path.exists(file_name  + '.keras'):
        model = keras.models.load_model(file_name + '.keras')
        
        with open(file_name + '.pkl', 'rb') as file:
            history = pickle.load(file)
        print(f"The file {file_name} already exists.")

    else:
        model = build_CNN(**params)
        X_train = tf.expand_dims(X_train, axis=-1)  # Assuming grayscale, add channel dimension
        X_test = tf.expand_dims(X_test, axis=-1)
        print('Training...')
        if train_datagen != None :
            history = model.fit(train_datagen.flow(X_train,Y_train,batch_size=64, seed=27,shuffle=False),
                    epochs=num_epochs,verbose=1, validation_data=(X_test, Y_test))
        else:
            history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size= batch_size, validation_data=(X_test, Y_test))
    
        model.save(file_name  + '.keras')
        # Save the training history separately
        with open(file_name + '.pkl', 'wb') as file:
            pickle.dump(history, file)
    
    # Retrieve the training metrics (after each train epoch) and the final test
    # accuracy.

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    test_accuracy = model.evaluate(x=X_test, y=Y_test, verbose=0,
                                    return_dict=True)['accuracy']
    title = 'cnn'
    for key in params.keys():
        title += '-' + key
    accuracy = AccuracyClass(train_accuracy, val_accuracy, num_epochs, title, model, test_accuracy )
    AccuracyList.append(accuracy)

    return test_accuracy