import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import numpy as np
import neuralNetwork as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def dataGenerate(X_train, X_test, Y_train, Y_test, layer_file='', testlayer_file='', input_model=None, layer_name='',feature_size=1024 ):
    train_features = None
    test_features = None

    if os.path.exists(layer_file) and os.path.exists(testlayer_file) :
        train_features =  pd.read_csv(layer_file)
        test_features = pd.read_csv(testlayer_file)

    else:
        #print(cnnrf.summary())
        layer_model = keras.Model(inputs=input_model.input,
                                    outputs=input_model.get_layer(layer_name).output)
        
        #layer_model.summary()
        print(X_train.shape)
        features = [layer_model.predict(np.expand_dims(X_train[i], axis=0))[0] for i in range(2000)]
        test_features = [layer_model.predict(np.expand_dims(X_test[i], axis=0))[0] for i in range(X_test.shape[0])]
        
        feature_col=[]
        for i in range(feature_size):
            feature_col.append("f_"+str(i))
            i+=1

        #Create DataFrame with features and coloumn name
        test_features=pd.DataFrame(data=test_features,columns=feature_col)
        train_features=pd.DataFrame(data=features,columns=feature_col)
        train_features.to_csv(layer_file)
        test_features.to_csv(testlayer_file)


    train_class = list(np.unique(Y_train))
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', Y_train[0:train_features.shape[0]].shape)
    print('Test Features Shape:', test_features.shape)
    print('Test Labels Shape:', Y_test[0:test_features.shape[0]].shape)
    return train_features, test_features

def train_and_evaluate(train_features, test_features, Y_train, Y_test, rf_file=''):
    #file_name= 'models/cnn-final20-rf_model.joblib'
    rf_classifier = None
    '''if os.path.exists(rf_file):
        rf_classifier = joblib.load(rf_file)
        print(f"The file {rf_file} already exists.")
    else:'''
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_features, Y_train[0:train_features.shape[0]])
        #joblib.dump(rf_classifier, rf_file)
    print(train_features.shape)
    print(test_features.shape)
    # Make predictions on the test set
    y_pred = rf_classifier.predict(test_features)
    # Evaluate the model
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(Y_test, y_pred))
    return rf_classifier

def plot_channel(activations, layer_names):
    
    # getting activations of each layer
    for idx, layer in enumerate(activations):
        if idx in (0,1,2,3):
            print('----------------')
            print('Geeting activations of layer',  idx+1, ':', layer_names[idx])
            activation = layer

            # shape of layer activation
            print('Images size is', activation.shape[1], 'x', activation.shape[2])
            print('Number of channels is', activation.shape[3])

            # print channels
            print('Printing channels:')
            
            # define nrows and ncols depending on number of channels
            if idx in (0,1):
                nrows, ncols = 4,8
            if idx in (2,3):
                nrows, ncols = 8,8

            # plots
            channel=0
            if idx in (0,1):
                f, axs = plt.subplots(nrows, ncols, figsize=(14,12))
            if idx in (2,3):
                f, axs = plt.subplots(nrows, ncols, figsize=(14,20))
                
            for i in range(nrows):
                for j in range(ncols):
                    if i==0 and j==0:
                        channel=0
                    else:
                        channel+=1

                    axs[i,j].matshow(activation[0,:, :, channel], cmap ='viridis')
                    axs[i,j].set(title=str(channel))
                    #axs[i,j].axis('off') # pay attention to the range of x and y axis
            plt.show()