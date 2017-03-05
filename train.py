import argparse
import csv
import cv2
import numpy as np
import os
import tensorflow as tf
import keras

from common import *
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

batch_size = 1024
nb_epoch = 8
keep_prob = 0.5


def load_data(path):
    """Load data from disk.

        :param path: path point to the directory that has the diriving_log.csv
        file and the IMG directory
        :return: return ndarray of center, left, right images and steering
        angles
        :rtype: tuple of 4 ndarray
    """
    x = []
    x_left = []
    x_right = []
    y = []

    with open('%s/driving_log.csv' % path, newline='') as csvfile:
        # (index to extract, list to persist to)
        image_extractor = [(0, x), (1, x_left), (2, x_right)]

        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            # 1. Load image to main memory
            for _, (index, collection) in enumerate(image_extractor):
                image_path = __rewrite_image_path(row[index], path)
                image = __load_image(image_path)
                collection.append(image)

            # 2. Get steering angle
            steering_angle = float(row[3])
            y.append(steering_angle)

    assert len(x) != 0
    assert len(x) == len(y)
    assert len(x) == len(x_left)
    assert len(x) == len(x_right)

    return np.array(x), np.array(x_left), np.array(x_right), np.array(y)

def __rewrite_image_path(source_path, path_rewrite):
    """ Rewrite the image path by extracting basename from **source_path** and
    using **dir_rewrite** as the directory
    """
    return '%s/IMG/%s' % (path_rewrite, os.path.basename(source_path))


def __load_image(path):
    """ Load an image to main memory

        :param path: image's path
        :return: ndarray containing the image data
        :rtype: ndarray
    """
    image = Image.open(path)
    # Resize image now to reduce main memory pressure
    image = resize_image(image)
    return np.asarray(image)


def get_model():
    """Get keras model of the steering net

        :return: Keras model
        :rtype: keras.models.model
    """
    model = Sequential()
    input_shape = (input_image_height, input_image_width, 3)
    # Normalization layer
    model.add(Lambda(lambda x: x / 255 - .5,\
            input_shape=input_shape,\
            output_shape=input_shape))
    # Layer 1: 5x5 convolution with 12 output filters
    model.add(Convolution2D(12, 5, 5, activation = 'relu', subsample=(2, 2)))
    # Layer 2: 3x3 convolution with 18 output filters
    model.add(Convolution2D(18, 3, 3, activation = 'relu', subsample=(2, 2)))
    # Layer 3: fully connected
    model.add(Flatten())
    # Layer 4: fully connected with 50 neurons
    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(keep_prob))
    # Layer 5: fully connected with 10 neurons
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(keep_prob))
    # Output
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


def save_model(model, path):
    """Save model to disk.

        :param model: model to save
        :param path: directory where to save the model
    """
    folder = "%s/outputs/" % path
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save model
    model_json = model.to_json()
    with open("%s/model.json" % folder, "w") as json_file:
        json_file.write(model_json)

    # Save weights
    model.save_weights("%s/model.h5" % folder)
    print("Saved model to disk")


def augment(center_images, left_images, right_images, steerings):
    """Augment the collection of given images and steerings.

        :param center_images: list of center images
        :param left_images: list of left images
        :param right_images: list of right images
        :param steerings: list of steering angles
        :return: augmented list of center images and steering angles
        :rtype: tuple of ndarray. First ndarray is the list of center images,
        second is the list of steering angles
    """
    i_augmented = []
    s_augmented = []
    for i in range(len(center_images)):
        __augment_image_trio(center_images[i], left_images[i],\
            right_images[i], steerings[i], i_augmented, s_augmented)
    return np.array(i_augmented), np.array(s_augmented)


def __augment_image_trio(center_image, left_image, right_image, steering,\
        out_images, out_steerings):
    """ Augment a trio of images.

        :param center_images: list of center images
        :param left_images: list of left images
        :param right_images: list of right images
        :param steerings: list of steering angles
        :param out_images: collection where images will be appended to
        :param out_steerings: collection where steering angles will be
        appended to
    """
    __augment_image_single(center_image, steering, out_images, out_steerings)

    # The idea was to use left and right image and infer the steering angle.
    # It had a tendency to deteriorate the final result instead of improving it.
    # Therefore, I decided to ignore left and right images.
    # offset = 0.05 if steering == 0 else steering/3.0
    #
    # left_steering = np.clip(steering + offset, 0, 1)
    # __augment_image_single(left_image, left_steering,\
    #     out_images, out_steerings)
    #
    # right_steering = np.clip(steering - offset, 0, 1)
    # __augment_image_single(right_image, right_steering,\
    #     out_images, out_steerings)


def __augment_image_single(image, steering, out_images, out_steerings):
    """ Augment a single image.

        :param image: image to augmente
        :param steering: steering angle corresponding to the image
        :param out_images: collection where images will be appended to
        :param out_steerings: collection where steering angles will be
        appended to
    """
    # Add original
    out_images.append(image)
    out_steerings.append(steering)

    # Add horizontal mirror
    out_images.append(cv2.flip(image, 1))
    out_steerings.append(-steering)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('data', type=str,
        help='Path to data used to train the model.')
    args = parser.parse_args()

    # 1. Load data
    X_center, X_left, X_right, y = load_data(args.data)
    loaded_count = len(X_center)
    print("Original data shape:", X_center[0].shape)

    # 2. Create model
    model = get_model()
    # Print out summary representation of model
    #model.summary()

    # 3. Data augmentation
    X_train, y_train = augment(X_center, X_left, X_right, y)
    gen_count = len(X_train) - loaded_count

    print("Number of examples =", loaded_count)
    print("Number of examples after data augmentation =", len(X_train))

    # 4. Split data in training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,\
        test_size=0.2, random_state=42)

    # 5. Setup real-time data augmentation
    datagen = ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05)
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)

    # 6. Train
    model.fit_generator(
        train_generator,
        samples_per_epoch=len(X_train),
        nb_epoch=nb_epoch,
        validation_data=(X_valid, y_valid),
        nb_val_samples=len(X_valid))

    # 7. Save
    save_model(model, args.data)
