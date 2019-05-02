from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from keras import backend as k
import cv2

def read_image(filename, img_shape):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
    (thresh, img) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    arr = np.array(img)
    # Normalize the values to be between 0 and 1. 255 is max value
    arr = (255 - arr)
    arr = arr.astype(float)
    arr /= 255

    return arr


def image_show(dataset, labels, num_to_show=8):
    rand_idx = np.random.choice(dataset.shape[0], num_to_show)
    images_and_labels = list(zip(dataset[rand_idx], labels[rand_idx]))

    img = plt.figure() #1, figsize=(15, 12), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(num_to_show/2.0), 2, index + 1)
        plt.axis('off')
        # each image is flat, we have to reshape to 2D array 28x28-784
        plt.imshow(image.reshape(28, 28), cmap='Greys',
                   interpolation='nearest')
        plt.title('Number: {}'.format(label.argmax()))
    
    plt.show()


def load_data(train_size, test_size, nb_classes):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    k.set_image_dim_ordering('th')

    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = x_test[:test_size]
    y_test = y_test[:test_size]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    x_train = x_train[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (x_train, y_train), (x_test, y_test)
