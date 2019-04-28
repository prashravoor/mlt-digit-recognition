from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sn
from utils import *
import pickle
from sklearn import metrics
import sys

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

    x_train = x_train.reshape(x_train.shape[0], input_shape)
    x_test = x_test.reshape(x_test.shape[0], input_shape)
    y_train = np.array([x.argmax() for x in y_train])
    y_test = np.array([x.argmax() for x in y_test])

    return (x_train, y_train), (x_test, y_test)

def grid_search(train_size, test_size):
    (x_train, y_train), (x_test, y_test) = load_dataset(
        train_size, test_size, nb_classes)
    
    svm = SVC()
    parameters = [{'kernel': ['poly'], 'gamma': [1e-3, 1e-2],
                       'C': [1, 10, 100], 'degree': range(1,5)}]
    print("Starting Grid Search")
    grid = GridSearchCV(svm, parameters, verbose=3)
    grid.fit(x_train, y_train) #grid search learning the best parameters
    print("Completed Grid Search. Best Parameters: ")
    print (grid.best_params_)

    return grid.best_estimator_

def svm_train(model=None, train_size=train_size, test_size=test_size):
    (x_train, y_train), (x_test, y_test) = load_dataset(
        train_size, test_size, nb_classes)

    if not model:
        svm = SVC(kernel='poly', gamma='scale', degree=2, C=10,
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
    print(pred)
    p_tmp = np.array(pred) < .8
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
    
    (x_train, y_train), (x_test, y_test) = load_dataset(10000, test_size, nb_classes)
    predicted = model.predict(x_test)
    confusion_matrix(model, y_test, predicted)

def predict_single(filename):
    img = read_image(
                filename, (28, 28)).reshape(input_shape)
    plt.imshow(img.reshape(28, 28), cmap='Greys')
    num = predict_class(model, img)
    plt.title('Number Predicted: {}'.format(num))
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
                    print('Invalid number specified for train size of test size!')
                    exit()
                if len(args) == 6 and args[5].lower() == 'search':
                    model = grid_search(tr_size, te_size)
            if not model:
                model = svm_train(train_size=tr_size,
                                test_size=te_size)
            else:
                model = svm_train(train_size=tr_size, test_size=te_size)
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
            show_confusion_matrix()
        else:
            print('Usage: {} {} {}'.format(
                args[0], 'train | predict', '[params]'))
            exit()
