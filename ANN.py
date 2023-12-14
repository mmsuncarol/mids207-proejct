
import numpy as np
import seaborn as sns  # for nicer plots
sns.set(style="darkgrid")  # default style

import tensorflow as tf
from tensorflow import keras
import os
import pickle
import neuralNetwork as nn


AccuracyList = []

def build_ANN(kernel_size=(5,5),strides=(1, 1),pool_size=(2, 2),learning_rate=0.001, cnn=False, hidden_cnn=[]):
  
    model = keras.models.Sequential()

    if cnn:
    # add first convolution layer to the model
        for i in range(len(hidden_cnn)):
            model.add(tf.keras.layers.Conv2D(filters=hidden_cnn[i],kernel_size=kernel_size,strides=strides,
                padding='same',data_format='channels_last',name=f'conv_{i}',activation='relu'))
        # add a max pooling layer with pool size and strides 
            model.add(tf.keras.layers.MaxPool2D(pool_size=pool_size,name=f'pool_{i}'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(6, activation='softmax'))

    # build model and print summary
    tf.random.set_seed(1)
    model.build(input_shape=(None, 128, 128, 1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy']) 
    
    return model

def train_and_evaluate(X_train, X_test, Y_train, Y_test, file_name='', num_epochs = 5, batch_size=32, params={}, train_datagen=None):
    
    model= None
    history = None

    if os.path.exists(file_name  + '.keras'):
        model = keras.models.load_model(file_name + '.keras')
        
        with open(file_name + '.pkl', 'rb') as file:
            history = pickle.load(file)
        print(f"The file {file_name} already exists.")

    else:
        tf.random.set_seed(1234)
        np.random.seed(1234)
        model = build_ANN(**params)
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
    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']

    test_accuracy = model.evaluate(x=X_test, y=Y_test, verbose=0,
                                    return_dict=True)['accuracy']

    accuracy = nn.AccuracyClass(train_accuracy, val_accuracy, num_epochs, file_name, model, test_accuracy, loss_values, val_loss_values)
    AccuracyList.append(accuracy)

    return test_accuracy