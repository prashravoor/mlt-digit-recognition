import svm
import time
import matplotlib.pyplot as plt

# Find time taken to train and predict varying sets of images using SVM

train_times = []
test_size = 100
ds_size = [100 , 200, 500, 1000, 5000, 10000, 20000, 30000, 40000, 60000]
for i in ds_size:
    start = time.time()
    svm.svm_train(train_size=i, test_size=test_size)
    end = time.time()
    train_times.append((i, end-start))

predict_times = []
svm.load_svm_model()
(x_train, y_train), (x_test, y_test) = svm.load_dataset(60000, 10000, 10)

pred_sizes = [50 , 100, 200, 500, 1000, 2000, 5000, 7500, 10000]
for i in pred_sizes:
    x_tmp = x_test[:i]
    start = time.time()
    for y in x_tmp:
        svm.predict_class(svm.model, y)
    end = time.time()
    predict_times.append((i, end-start))

train_times = list(zip(*train_times))
plt.plot(train_times[0], train_times[1])
plt.title('Training Times for SVM')
plt.xlabel('Number of images')
plt.ylabel('Time (s)')
plt.show()

predict_times = list(zip(*predict_times))
plt.plot(predict_times[0], predict_times[1])
plt.title('Prediction Times for SVM')
plt.xlabel('Number of images')
plt.ylabel('Time (s)')
plt.show()




