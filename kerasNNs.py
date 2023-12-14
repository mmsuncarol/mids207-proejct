import pickle
import os
import tensorflow as tf
from tensorflow import keras


def train_and_evaluate(X_train, X_test, Y_train, Y_test, file_name='', num_epochs = 1, batch_size=32, kmodel=None, classSize=6):
    head_model = None
    history = None
    #file_name= 'models/inception'
    if os.path.exists(file_name + '.keras'):
        head_model = keras.models.load_model(file_name + '.keras')

        with open(file_name + '.pkl', 'rb') as file:
            history = pickle.load(file)
        print(f"The file {file_name} already exists.")
    
    else:
        # Train the model
        head_model = keras.Sequential()
        #head_model.add(tf.keras.applications.InceptionV3(include_top=False, pooling='max'))
        head_model.add(kmodel)
        head_model.add(keras.layers.Dense(classSize, activation='softmax'))

        for layer in head_model.layers[:]:
            layer.trainable = True

        #head_model = keras.Model(inputs = base_model.input,  outputs = predictions)
        #head_model = keras.Sequential(base_model + predictions)
        head_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        X_trainRes = tf.expand_dims(X_train.copy(), axis=3)  # Assuming grayscale, add channel dimension
        X_testRes = tf.expand_dims(X_test.copy(), axis=3)
        X_trainRes = tf.repeat(X_trainRes, 3, axis=3)
        X_testRes = tf.repeat(X_testRes, 3, axis=3)

        #history = head_model.fit(train_datagen.flow(X_trainRes,Y_train,batch_size=128, seed=27,shuffle=False), epochs=1,verbose=1, validation_data=(X_testRes, Y_test))
        #history = head_model.fit(train_datagen.flow(X_trainRes,Y_train,batch_size=128, seed=27,shuffle=False),epochs=1,verbose=1, validation_data=(X_testRes, Y_test))
        history = head_model.fit(X_trainRes, Y_train, epochs=num_epochs, batch_size= batch_size, validation_data=(X_testRes, Y_test))
        head_model.save(file_name + '.keras')
        with open(file_name + '.pkl', 'wb') as file:
                pickle.dump(history, file)

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    print(f"{file_name}  Test Accuracy:{train_accuracy} {val_accuracy}")
    return head_model