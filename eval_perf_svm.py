import svm
import cnn
import time
import matplotlib.pyplot as plt

# Find time taken to train and predict varying sets of images using SVM
train_times = []
train_times_c = []
test_size = 100
ds_size = [100, 200, 500, 1000, 5000, 10000, 20000, 30000, 40000, 60000]
for i in ds_size:
    start = time.time()
    svm.svm_train(train_size=i, test_size=test_size)
    end = time.time()
    train_times.append((i, end-start))

    start = time.time()
    cnn.cnn_train(train_size=i, test_size=test_size, show_graph=False)
    end = time.time()
    train_times_c.append((i, end-start))

predict_times = []
predict_times_c = []

svm.load_svm_model()
(x_train, y_train), (x_test, y_test) = svm.load_dataset(60000, 10000, 10)

cnn.load_cnn_model()
(x_train_c, y_train_c), (x_test_c, y_test_c) = cnn.load_data(60000, 10000, 10)

pred_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 7500, 10000]
for i in pred_sizes:
    x_tmp = x_test[:i]

    start = time.time()
    for y in x_tmp:
        svm.predict_class(svm.model, y)
    end = time.time()
    predict_times.append((i, end-start))

    start = time.time()
    for y in x_tmp:
        cnn.predict_class(cnn.model, y)
    end = time.time()
    predict_times_c.append((i, end-start))

train_times = list(zip(*train_times))
train_times_c = list(zip(*train_times_c))
plt.plot(train_times[0], train_times[1])
plt.plot(train_times_c[0], train_times_c[1])
plt.title('Training Times for SVM vs CNN')
plt.xlabel('Number of images')
plt.ylabel('Time (s)')
plt.legend(['SVM', 'CNN'])
plt.show()


predict_times = list(zip(*predict_times))
predict_times_c = list(zip(*predict_times_c))
plt.plot(predict_times[0], predict_times[1])
plt.plot(predict_times_c[0], predict_times_c[1])
plt.title('Prediction Times for SVM vs CNN')
plt.xlabel('Number of images')
plt.ylabel('Time (s)')
plt.legend(['SVM', 'CNN'])
plt.show()
