import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.core import Activation
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from utils import *
from collections import Counter
import sys
import seaborn as sn

nb_epoch = 5
batch_size = 128
optimizer = 'adam'
validation_split = .2
nb_classes = 10
input_shape = (1, 28, 28)

layer1_count = 28
layer2_count = 50
layer3_count = 500
kernel_size = 5
pool_size = (2, 2)

model = None
train_size = 100
test_size = 50


def build_model(layer1_count, kernel_size, input_shape, pool_size, num_classes,
                layer2_count, layer3_count):
    model = Sequential()

    model.add(Conv2D(layer1_count, kernel_size=kernel_size,
                     padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_size))

    model.add(Conv2D(layer2_count, kernel_size=kernel_size, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, strides=pool_size))
    # model.add(Dropout(0.2))

    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense(layer3_count))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def train(model, x_train, y_train, batch_size, num_epochs):
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        verbose=1,
                        validation_split=validation_split,
                        epochs=num_epochs)

    return history


def predict_class(model, data):
    pred = model.predict(data.reshape((1, 1, 28, 28)))
    p_tmp = pred < .6
    pred[p_tmp] = -1
    if max(pred[0]) == -1:
        return 'Not a number'
    return pred.argmax()


def predict_single(filename):
    img = read_image(
        filename, (input_shape[1], input_shape[2])).reshape(input_shape)
    plt.imshow(img.reshape(28, 28), cmap='Greys')
    num = predict_class(model, img)
    plt.title('Number Predicted: {}'.format(num))
    plt.show()


def evaluate(model, x_test, y_test):
    score = model.evaluate(x_test, y_test)
    return score


def confusion_matrix(model, labels, predicted):
    pred_conv = np.array([x.argmax() for x in predicted])
    labels_conv = np.array([x.argmax() for x in labels])
    cm = metrics.confusion_matrix(labels_conv, pred_conv)
    print(cm)
    # plt.matshow(cm, cmap='Greys')
    sn.heatmap(cm, annot=True)
    plt.show()


def show_confusion_matrix(test_size=test_size):
    if not model:
        load_cnn_model()
    if not model:
        print('You need to train the model first!')
        return

    if test_size < 50 or test_size > 10000:
        print('Invalid size specified for test size, will use default 1000')
        test_size = 1000

    (x_train, y_train), (x_test, y_test) = load_data(10000, test_size, nb_classes)
    predicted = model.predict(x_test)
    confusion_matrix(model, y_test, predicted)


def plot_graphs(history):
    # print(history.history.keys())
    _, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['acc'])
    ax[0].plot(history.history['val_acc'])
    ax[0].set_title('Accuracy Trace')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(['Train', 'Test'])

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Loss Function Trace')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Error')
    ax[1].legend(['Train', 'Test'])
    plt.show()


def cnn_train(train_size=train_size, test_size=test_size,
              layer1_count=layer1_count, layer2_count=layer2_count,
              layer3_count=layer3_count, kernel_size=kernel_size,
              input_shape=input_shape,
              pool_size=(2, 2), num_classes=nb_classes,
              batch_size=batch_size, num_epochs=nb_epoch,
              show_graph=True):

    if train_size < 100 or train_size > 60000:
        print('Invalid size specified for train size, will use default 10000')
        train_size = 10000
    if test_size < 50 or test_size > 10000:
        print('Invalid size specified for test size, will use default 1000')
        test_size = 1000

    (x_train, y_train), (x_test, y_test) = load_data(
        train_size, test_size, num_classes)
    # image_show(x_train, y_train)
    print('Training Data Size: {}, Test Data Size: {}'.format(
        x_train.shape, x_test.shape))
    model = build_model(layer1_count=layer1_count,
                        kernel_size=kernel_size, input_shape=input_shape,
                        pool_size=pool_size, num_classes=num_classes,
                        layer2_count=layer2_count, layer3_count=layer3_count)
    history = train(model, x_train, y_train, batch_size, num_epochs)
    score = evaluate(model, x_test, y_test)
    print('Accuracy: {}, Score: {}'.format(score[1], score[0]))
    if show_graph:
        plot_graphs(history)
    return model, history


def save_model(filename='cnn.hdf5'):
    model.save(filename)


def load_cnn_model(filename='cnn.hdf5'):
    global model
    model = load_model(filename)


if __name__ == '__main__':

    args = sys.argv
    if len(args) == 1:
        print('Training and saving the model to file cnn.hdf5')
        model, _ = cnn_train()
        save_model()
    else:
        if str(args[1]).lower() == 'train':
            file = None
            if len(args) > 2:
                file = args[2]
                print('Training the model and saving to file {}'.format(file))
            else:
                print('Training and saving the model to file cnn.hdf5')
                model, _ = cnn_train()
                save_model()
                exit()

            nepochs = nb_epoch
            tr_size = train_size
            te_size = test_size
            if len(args) >= 5:
                if len(args) == 6:
                    print('Will Train for {} Epochs, Using {} Training Images, {} Test Images'.format(
                        args[5], args[3], args[4]
                    ))
                    try:
                        nepochs = int(args[5])
                    except:
                        print('Invalid number specified for number of epochs!')
                        exit()
                else:
                    print('Will Train for {} Epochs, Using {} Training Images, {} Test Images'.format(
                        nepochs, args[3], args[4]
                    ))
                try:
                    tr_size = int(args[3])
                    te_size = int(args[4])
                except:
                    print('Invalid number specified for train size of test size!')
                    exit()

            model, _ = cnn_train(train_size=tr_size,
                                 test_size=te_size, num_epochs=nepochs)
            save_model(file)
        elif str(args[1]).lower() == 'predict':
            if not len(args) >= 3:
                print('Usage: {} {} {} {}'.format(
                    args[0], args[1], '<path to image file>', '[Model File]'))
                exit()
            if len(args) == 4:
                print('Loading Model from file {}'.format(args[3]))
                load_cnn_model(args[3])
            else:
                load_cnn_model()
            print('Model loaded')

            print('Will predict class of digit in file {}'.format(args[2]))
            predict_single(args[2])

        elif str(args[1]).lower() == 'confusion':
            te_size = 1000
            if len(args) == 3:
                try:
                    te_size = int(args[2])
                except:
                    print('Invalid size for Confusion matrix, will use default 1000')
            show_confusion_matrix(te_size)
        else:
            print('Usage: {} {} {}'.format(
                args[0], 'train | predict', '[params]'))
            exit()
