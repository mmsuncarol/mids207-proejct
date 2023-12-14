# Import the libraries we'll use below.
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns  # for nicer plots
sns.set(style="darkgrid")  # default style
import os

import tensorflow as tf
from tensorflow import keras
from keras import metrics
import pickle

tf.get_logger().setLevel('INFO')

AccuracyList = []

def build_model(n_classes,
                hidden_layer_sizes=[],
                activation='relu',
                optimizer='SGD',
                learning_rate=0.01):
  """Build a multi-class logistic regression model using Keras.

  Args:
    n_classes: Number of output classes in the dataset.
    hidden_layer_sizes: A list with the number of units in each hidden layer.
    activation: The activation function to use for the hidden layers.
    optimizer: The optimizer to use (SGD, Adam).
    learning_rate: The desired learning rate for the optimizer.

  Returns:
    model: A tf.keras model (graph).
  """
  tf.keras.backend.clear_session()
  np.random.seed(0)
  tf.random.set_seed(0)

  model = tf.keras.Sequential()
  model.add(keras.layers.Flatten())
  # Set input shape in advance
  #model.add(tf.keras.Input(shape=(n_classes,), name='Input'))
  for hidden_layer_size in hidden_layer_sizes:
    model.add(tf.keras.layers.Dense(units=hidden_layer_size,activation=activation, input_shape=(None, 128, 128, 1)))
    
  model.add(keras.layers.Dense(units=n_classes, activation="softmax", name='Output', input_shape=(None, 128, 128, 1)))

  model.build(input_shape=(None, 128, 128, 1))
  # Create an instance of the optimizer with the specified learning rate
  print(model.summary())
  opt =tf.keras.optimizers.SGD(learning_rate=learning_rate)
  #print(optimizer)
  if optimizer != 'SGD':
    opt =tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

  return model

class AccuracyClass2:
  def __init__(self, test):
    self.test = test
    
class AccuracyClass:
  def __init__(self, train_accuracy , val_accuracy, num_epochs, title, model, test_accuracy, loss_values = None, val_loss_values =None):
    self.train_accuracy = train_accuracy
    self.val_accuracy = val_accuracy
    self.num_epochs = num_epochs
    self.title = title
    self.model = model
    self.test_accuracy = test_accuracy
    self.loss_values = loss_values
    self.val_loss_values = val_loss_values
  
  def plot(self):
    plt.plot(self.train_accuracy, label='train_accuracy')
    plt.plot(self.val_accuracy, label='validation accuracy')
    if self.loss_values != None:
      plt.plot(self.loss_values, label='loss values')
    if self.val_loss_values != None:
      plt.plot(self.val_loss_values, label='val loss values')
    plt.xticks(range(self.num_epochs))
    plt.xlabel('Train epochs')
    plt.title(self.title)
    plt.legend()
    plt.show()

  def plot_accuracy(self):
    plt.plot(self.train_accuracy, label='train_accuracy')
    plt.plot(self.val_accuracy, label='validation accuracy')
    plt.xticks(range(self.num_epochs))
    plt.xlabel('Train epochs')
    plt.title(self.title)
    plt.legend()
    plt.show()

  def plot_loss(self):
    if self.loss_values != None:
      plt.plot(self.loss_values, label='loss values')
    if self.val_loss_values != None:
      plt.plot(self.val_loss_values, label='val loss values')
    plt.xticks(range(self.num_epochs))
    plt.xlabel('Train epochs')
    plt.title(self.title)
    plt.legend()
    plt.show()


def train_and_evaluate(X_train, X_test, Y_train, Y_test,
                       hidden_layer_sizes=[],
                       activation='tanh',
                       optimizer='Adam',
                       learning_rate=0.01,
                       num_epochs=3, file_name='', train_datagen=None):

  # Build the model.


  model = None
  history = None

  if os.path.exists(file_name + '.keras'):
    model = keras.models.load_model(file_name + '.keras')
    with open(file_name + '.pkl', 'rb') as file:
      history = pickle.load(file)
    print(f"The file {file_name} already exists.")
  
  else:
    model = build_model(n_classes=6,
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    optimizer=optimizer,
                    learning_rate=learning_rate)
    # Train the model.
    for layer in model.layers:
      print(layer.input_shape)
    print('Training...')
    if train_datagen != None :
      X_train = tf.expand_dims(X_train, axis=-1)
      X_test = tf.expand_dims(X_test, axis=-1)
      newTrain = train_datagen.flow(X_train,Y_train,batch_size=64, seed=27,shuffle=False)
      print(newTrain)
      history = model.fit(train_datagen.flow(X_train,Y_train,batch_size=64, seed=27,shuffle=False),
                    epochs=num_epochs,verbose=1, validation_data=(X_test, Y_test))
      '''model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    epochs=epochs, # one forward/backward pass of training data
                    steps_per_epoch=x_train.shape[0]//batch_size, # number of images comprising of one epoch
                    validation_data=(x_test, y_test), # data for validation
                    validation_steps=x_test.shape[0]//batch_size)
                    '''
    
    else:
      print(X_train.shape, Y_train.shape)
      history = model.fit(x=X_train,y=Y_train,epochs= num_epochs,batch_size=64,validation_split=0.1,verbose=1)
    
    
    model.save(file_name + '.keras')
    # Save the training history separately
    with open(file_name + '.pkl', 'wb') as file:
        pickle.dump(history, file)

  # Retrieve the training metrics (after each train epoch) and the final test
  # accuracy.
  train_accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  test_accuracy = model.evaluate(x=X_test, y=Y_test, verbose=0,
                                 return_dict=True)['accuracy']
  
  title = 'layer=' + str(hidden_layer_sizes) + ' activation=' + activation + ' optimizer=' + optimizer
  accuracy = AccuracyClass(train_accuracy, val_accuracy, num_epochs, title, model, test_accuracy )
  AccuracyList.append(accuracy)

  #print(model.summary())
  return test_accuracy