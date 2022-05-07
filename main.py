import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from math import floor

# Keras
from keras import Model
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD, Adam
from keras.metrics import Accuracy
from keras.applications.mobilenet_v2 import MobileNetV2

BATCH_SIZE = 32
SAMPLES = 25000
TRAIN_SIZE = floor((SAMPLES / BATCH_SIZE) * 0.8)


def load_data(directory_name):
    return tf.keras.utils.image_dataset_from_directory(directory=directory_name, label_mode="binary", crop_to_aspect_ratio=True, batch_size=BATCH_SIZE)


def process_data(features, targets):
    return features / 255.0, targets


def explore_data(dataset: tf.data.Dataset):
    """Explore The Data By Plotting The Images Printing The Shape"""
    SAMPLE_SIZE = 3
    fig, axs = plt.subplots(SAMPLE_SIZE, 1)
    dataset = dataset.take(1).as_numpy_iterator()
    img_batch, label_batch = list(dataset)[0]
    for img, label, ax in zip(img_batch[0:SAMPLE_SIZE], label_batch[0:SAMPLE_SIZE], axs):
        ax.imshow(np.array(img))
    plt.show()


def define_model_1():
    """Define A Keras Model Which Exhibits Overfitting"""
    inputs = Input(shape=(256, 256, 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same")(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same")(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same")(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    flatten = Flatten()(pool3)
    dense1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    outputs = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=inputs, outputs=outputs, name="cats_vs_dogs")
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[Accuracy().name])
    return model


def define_model_2():
    """Define A Keras Model With Dropout Regularization To Avoid Over-Fitting"""
    inputs = Input(shape=(256, 256, 3))

    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same")(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    drop1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same")(drop1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    drop2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same")(drop2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    drop3 = Dropout(0.2)(pool3)

    flatten = Flatten()(drop3)

    dense = Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)
    drop4 = Dropout(0.2)(dense)

    outputs = Dense(1, activation='sigmoid')(drop4)

    model = Model(inputs=inputs, outputs=outputs, name="cats_vs_dogs")
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[Accuracy().name])
    return model


def define_model_3():
    """Define A Keras Model Using A Pre-Trained MobileNetV2 Encoder"""
    mobile_net = MobileNetV2(include_top=False, input_shape=(256, 256, 3))
    for layer in mobile_net.layers:
        layer.trainable = False

    flatten = Flatten()(mobile_net.layers[-1].output)

    dense = Dense(128, activation='relu', kernel_initializer='he_uniform')(flatten)

    outputs = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=mobile_net.inputs, outputs=outputs, name="cats_vs_dogs")
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[Accuracy().name])
    return model


def main():
    # Load Data
    raw_data: tf.data.Dataset = load_data('train').map(process_data)
    explore_data(raw_data)

    # Split Training And Test Data
    training_data = raw_data.take(625)
    test_data = raw_data.skip(625)

    # Define Model
    model = define_model_2()
    model.summary()

    # Train Model
    model.fit(training_data, epochs=5, verbose=1, validation_data=test_data)


if __name__ == '__main__':
    main()
