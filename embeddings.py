
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

def build_embeddings_model(average_over_positions=False,
                           cat_size=6,
                           sequence_length=16384,
                           embedding_dim=6):
  
  """Build a tf.keras model using embeddings."""
  # Clear session and remove randomness.
  tf.keras.backend.clear_session()
  tf.random.set_seed(0)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Embedding(
      input_dim=cat_size,
      output_dim=5,
      input_length=sequence_length)
  )

  if average_over_positions:
    # This layer averages over the first dimension of the input by default.
    model.add(tf.keras.layers.GlobalAveragePooling1D())
  else:
    # Concatenate.
    model.add(tf.keras.layers.Flatten())
    
  model.add(tf.keras.layers.Dense(
      units=cat_size,                     # output dim (for binary classification)
      activation='softmax'         # apply the softmax function!
  ))

  model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])

  return model

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

def train_and_evaluate(X_train, X_test, Y_train, Y_test,
                       cat_size = 6,
                       num_epochs=5, file_name=''):
  

    
    if os.path.exists(file_name + '.keras'):
        model = keras.models.load_model(file_name + '.keras')
        with open(file_name + '.pkl', 'rb') as file:
            history = pickle.load(file)
        print(f"The file {file_name} already exists.")

    else:
        sequence = int(X_train.shape[1]) * int(X_train.shape[2])
        model = build_embeddings_model(cat_size=cat_size,
                               sequence_length=sequence,
                               embedding_dim=2)
        print('Training...')

        X_flattened = X_train.reshape(X_train.shape[0], int(X_train.shape[1]) * int(X_train.shape[2]))

        print(X_flattened.shape)
        print(list(set(Y_train)))

        history = model.fit(
        x=X_flattened,
        y=Y_train,
        epochs= num_epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=0)
    
    model.save(file_name + '.keras')
    # Save the training history separately
    with open(file_name + '.pkl', 'wb') as file:
        pickle.dump(history, file)
    
    # Retrieve the training metrics (after each train epoch) and the final test
    # accuracy.

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    X_flat_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]) * int(X_test.shape[2]))
    test_accuracy = model.evaluate(x=X_flat_test, y=Y_test, verbose=0,
                                    return_dict=True)['accuracy']
    title = 'Embeddings'
    accuracy = AccuracyClass(train_accuracy, val_accuracy, num_epochs, title, model, test_accuracy )
    AccuracyList.append(accuracy)

    return test_accuracy