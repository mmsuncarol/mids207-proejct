import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers
import neuralNetwork as nn
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os, pickle
from keras.models import model_from_json
import pandas as pd

AccuracyList = []

def mlp(x, hidden_units, dropout_rate):
    #print(x)
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size


    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        #print(patches.shape)
        return patches

class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )


    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(72, 72),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

def create_vit_classifier(input_shape, patch_size=6, num_patches=144, projection_dim=64, transformer_layers = 4, num_heads=4,num_classes=6):
    input_shape = (128, 128, 3)
    
    #print(input_shape)
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    #print(data_augmentation.layers)
    augmented = data_augmentation(inputs)
    # Create patches.
    #print(augmented.shape)
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]  # Size of the transformer layers
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])


    # Create a [batch_size, projection_dim] tensor.
    #print(encoded_patches)
    #representation = layers.Input(shape=input_shape)
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Add MLP.
    mlp_head_units = [2048, 1024]
    #representation = tf.keras.layers.Input(shape=input_shape)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    #print(features.shape)
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model

def train_and_evaluate(X_train, X_test, Y_train, Y_test, file_name='', num_epochs = 5, patch_size=6 ,batch_size=5, num_patches=144, learning_rate=0.001, weight_decay=0.0001):
    
    #X_t= np.expand_dims(X_train.copy(), axis=-1)
    X_train = tf.expand_dims(X_train.copy(), axis=3)
    X_train = tf.repeat(X_train, 3, axis=3)
    X_test = tf.expand_dims(X_test.copy(), axis=3)
    X_test = tf.repeat(X_test, 3, axis=3)
    
    checkpoint_filepath = file_name + ".weights.h5"
    data_augmentation.layers[0].adapt(X_train)
    
    '''
    image_size = 72
    plt.figure(figsize=(4, 4))
    image = X_train[np.random.choice(range(X_train.shape[0]))]
    plt.imshow(image)
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )
    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))

    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")
    '''

    hist_df = None
    vit_classifier = create_vit_classifier(input_shape=(X_train.shape[1], X_train.shape[2], 3), 
                                           patch_size=patch_size, num_patches=num_patches, projection_dim=64, 
                                           transformer_layers = 4, num_heads=4,num_classes=6)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    vit_classifier.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    if os.path.exists(checkpoint_filepath):
        with open(file_name + '-history.csv' , mode='r') as f:
            hist_df = pd.read_csv(f)

        print(f"The file {file_name} already exists.")
       
    else:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        history = vit_classifier.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=num_epochs,validation_split=0.1,
                                     callbacks=[checkpoint_callback],)

        hist_df = pd.DataFrame(history.history) 
        with open(file_name + '-history.csv' , mode='w') as f:
            hist_df.to_csv(f)

    vit_classifier.load_weights(checkpoint_filepath)

    _, accuracy, top_5_accuracy = vit_classifier.evaluate(X_test, Y_test)
    #print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 3)}%")

    train_accuracy = hist_df['accuracy']
    val_accuracy = hist_df['val_accuracy']

    title = 'ViT'
    vaccuracy = nn.AccuracyClass(train_accuracy, val_accuracy, num_epochs, title, vit_classifier, accuracy )
    AccuracyList.append(vaccuracy)

    return train_accuracy, val_accuracy, vit_classifier
