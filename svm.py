from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sn
from utils import *
import pickle
from sklearn import metrics
import sys
import math
from matplotlib import gridspec

train_size = 300
test_size = 100
nb_classes = 10
input_shape = 784  # 28 * 28

model = None


def save_model(filename='svm.dump'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
        f.close()


def load_svm_model(filename='svm.dump'):
    global model
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        f.close()


def load_dataset(train_size, test_size, num_classes):
    (x_train, y_train), (x_test, y_test) = load_data(
        train_size, test_size, nb_classes)

    if train_size < 100 or train_size > 60000:
        print('Invalid size specified for train size, will use default 10000')
        train_size = 10000
    if test_size < 50 or test_size > 10000:
        print('Invalid size specified for test size, will use default 1000')
        test_size = 1000

    x_train = x_train.reshape(x_train.shape[0], input_shape)
    x_test = x_test.reshape(x_test.shape[0], input_shape)
    y_train = np.array([x.argmax() for x in y_train])
    y_test = np.array([x.argmax() for x in y_test])

    return (x_train, y_train), (x_test, y_test)


def grid_search(train_size, test_size):
    (x_train, y_train), (x_test, y_test) = load_dataset(
        train_size, test_size, nb_classes)

    svm = SVC()
    parameters = [{'kernel': ['poly', 'rbf', 'linear'], 'gamma': [1e-3, 1e-2, .1],
                   'C': [1, 10, 100], 'degree': range(1, 5)}]
    print("Starting Grid Search")
    grid = GridSearchCV(svm, parameters, verbose=3, n_jobs=4)
    grid.fit(x_train, y_train)  # grid search learning the best parameters
    print("Completed Grid Search. Best Parameters: ")
    print(grid.best_params_)

    return grid.best_estimator_


def svm_train(model=None, train_size=train_size, test_size=test_size):
    (x_train, y_train), (x_test, y_test) = load_dataset(
        train_size, test_size, nb_classes)

    if not model:
        svm = SVC(kernel='rbf', gamma=.01, degree=1, C=10,
                  probability=True, random_state=42)
    else:
        svm = model

    print("Starting Model Training")
    svm.fit(x_train, y_train)

    print("Computing Accuracy")
    predicted = svm.score(x_test, y_test)
    print('Score: {}'.format(predicted))
    return svm


def predict_class(model, data):
    pred = model.predict_proba(data.reshape(1, input_shape))
    p_tmp = np.array(pred) < .5
    arr = np.array(pred)
    arr[p_tmp] = -1
    if max(arr[0]) == -1:
        return 'Not a Number'
    return arr.argmax()


def confusion_matrix(model, labels, predicted):
    pred_conv = np.array(predicted)
    labels_conv = np.array(labels)
    cm = metrics.confusion_matrix(labels_conv, pred_conv)
    print(cm)
    # plt.matshow(cm, cmap='Greys')
    sn.heatmap(data=cm, annot=True)
    plt.show()


def show_confusion_matrix(test_size=test_size):
    if not model:
        load_svm_model()

    if not model:
        print('You need to train the model first!')
        return

    if test_size < 50 or test_size > 10000:
        print('Invalid size specified for test size, will use default 1000')
        test_size = 1000

    (x_train, y_train), (x_test, y_test) = load_dataset(
        10000, test_size, nb_classes)
    predicted = model.predict(x_test)
    confusion_matrix(model, y_test, predicted)


def predict_single(filename):
    img = read_image(
        filename, (28, 28)).reshape(input_shape)
    plt.imshow(img.reshape(28, 28), cmap='Greys', interpolation='nearest')
    num = predict_class(model, img)
    plt.title('Number Predicted: {}'.format(num))
    plt.axis('off')
    plt.show()


def predict_multiple(filenames):
    if len(filenames) == 1:
        predict_single(filenames[0])
    else:
        cols = 5
        rows = math.ceil(len(filenames)/5.0)
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure()
        n = 0
        for file in filenames:
            img = read_image(file, (28, 28)).reshape(input_shape)
            ax = fig.add_subplot(gs[n])
            n += 1
            print(n)
            num = predict_class(model, img)
            ax.imshow(img.reshape(28, 28), cmap='Greys',
                      interpolation='nearest')
            ax.set_title('Number Predicted: {}'.format(num))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print('Training and saving the model to file svm.dump')
        model = svm_train()
        save_model()
    else:
        if str(args[1]).lower() == 'train':
            file = None
            if len(args) > 2:
                file = args[2]
                print('Training the model and saving to file {}'.format(file))
            else:
                print('Training and saving the model to file svm.dump')
                model = svm_train()
                save_model()
                exit()

            tr_size = train_size
            te_size = test_size
            if len(args) >= 5:
                print('Will Train Using {} Training Images, {} Test Images'.format(
                    args[3], args[4]
                ))
                try:
                    tr_size = int(args[3])
                    te_size = int(args[4])
                except:
                    print('Invalid number specified for train size or test size!')
                    exit()
                if len(args) == 6 and args[5].lower() == 'search':
                    model = grid_search(tr_size, te_size)

            model = svm_train(train_size=tr_size,
                              test_size=te_size)
            save_model(file)
        elif str(args[1]).lower() == 'predict':
            if not len(args) >= 3:
                print('Usage: {} {} {} {}'.format(
                    args[0], args[1], '<path to image file>', '[Model File]'))
                exit()

            if len(args) == 4:
                print('Loading Model from file {}'.format(args[3]))
                load_svm_model(args[3])
            else:
                load_svm_model()
            print('Model loaded')

            print('Will predict class of digit in file {}'.format(args[2]))
            predict_single(args[2])

        elif str(args[1]).lower() == 'confusion':
            te_size = 1000
            if len(args) == 3:
                try:
                    te_size = int(args[2])
                except:
                    print(
                        'Invalid size specified for test dataset, will use default 1000')
            show_confusion_matrix(te_size)
        elif args[1].lower() == 'showrand':
            if not model:
                load_svm_model()
                if not model:
                    print('You need to train the model first!')
                    exit()

            (x, y), (p, q) = load_dataset(30000, 10000, 10)
            cols = 4
            rows = 2
            gs = gridspec.GridSpec(rows, cols)
            fig = plt.figure()
            n = 0
            for _ in range(8):
                ax = fig.add_subplot(gs[n])
                n += 1
                z = np.random.randint(0, len(x))
                num = predict_class(model, x[z])
                ax.imshow(x[z].reshape(28, 28), cmap='Greys')
                ax.set_title('Number Predicted: {}'.format(num))
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
        else:
            print('Usage: {} {} {}'.format(
                args[0], 'train | predict', '[params]'))
            exit()
