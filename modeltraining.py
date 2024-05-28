# dataset_path = "traffic_sign_classification_dataset\\train"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import cv2
import numpy as np

# Set the path to your dataset
dataset_path = "path to dataset"

# Parameters
input_shape = (224, 224, 3)  # Adjusted input shape to include channel dimension
num_classes = len(os.listdir(dataset_path))
epochs = 10  # Specify the number of epochs

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# Split the dataset into training and validation sets
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Resize images to (224, 224)
    batch_size=32,
    class_mode="categorical",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # Resize images to (224, 224)
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_vgg19_model(input_shape, num_classes):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape)
    )
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model


# Create the VGG19 model
model = create_vgg19_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model with tqdm progress bar for epochs
for epoch in tqdm(range(epochs), desc="Training"):
    # Train the model for ten epoch
    history = model.fit(train_generator, epochs=50, validation_data=val_generator)

# Save the trained model
model.save("vgg19_sign_lang_model.h5")
