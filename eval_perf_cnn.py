import cnn
import matplotlib.pyplot as plt

train_size = 50
test_size = 20

# Evaluate different parameters, keeping the rest as constant
kernel_size = 5
layer1_count = 28
layer2_count = 50
layer3_count = 500
batch_size = 128
epochs = 5
pool_size = 2


def plot_graph(data, title, x, y):
    for z in data:
        plt.plot(z[1])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend([z[0] for z in data])
    plt.show()

# Number of epochs
e_data = []
for i in range(2,50):
    _, history = cnn.cnn_train(train_size=train_size, test_size=test_size,
    kernel_size=kernel_size, pool_size=(pool_size, pool_size),
     num_epochs=i,
    layer2_count=layer2_count, layer3_count=layer3_count,
    layer1_count=i, batch_size=batch_size, show_graph=False)
    e_data.append((i, history.history['acc']))


# Layer 1 Neurons
l1 = [2**x for x in range(6)]
l1.append(28) # Size of the images
l1_data = []
for i in l1:
    _,history = cnn.cnn_train(train_size=train_size, test_size=test_size,
    kernel_size=kernel_size, pool_size=(pool_size, pool_size), num_epochs=epochs,
    layer2_count=layer2_count, layer3_count=layer3_count,
    layer1_count=i, batch_size=batch_size, show_graph=False)
    l1_data.append((i, history.history['acc']))

# Layer 3 Neurons
l2 = [10*x for x in range(1,10)]
l2_data = []
for i in l2:
    _,history = cnn.cnn_train(train_size=train_size, test_size=test_size,
    kernel_size=kernel_size, pool_size=(pool_size, pool_size), num_epochs=epochs,
    layer2_count=i, layer3_count=layer3_count,
    layer1_count=layer1_count, batch_size=batch_size, show_graph=False)
    l2_data.append((i, history.history['acc']))


# Layer 3 Neurons
l3 = [2**x for x in range(6,11)] # Powers of 2 from 64 to 1024
l3_data = []
for i in l3:
    _,history = cnn.cnn_train(train_size=train_size, test_size=test_size,
    kernel_size=kernel_size, pool_size=(pool_size, pool_size), num_epochs=epochs,
    layer2_count=layer2_count, layer3_count=i,
    layer1_count=layer1_count, batch_size=batch_size, show_graph=False)
    l3_data.append((i, history.history['acc']))

# Kernel Size
k = range(2, 16) # 2 - 15
k_data = []
for i in k:
    _,history = cnn.cnn_train(train_size=train_size, test_size=test_size,
    kernel_size=i, pool_size=(pool_size, pool_size), num_epochs=epochs,
    layer2_count=layer2_count, layer3_count=layer3_count,
    layer1_count=layer1_count, batch_size=batch_size, show_graph=False)
    k_data.append((i, history.history['acc']))

# Batch Size
b = [2**x for x in range(1, 11)] # 2 - 1024
b_data = []
for i in b:
    _,history = cnn.cnn_train(train_size=train_size, test_size=test_size,
    kernel_size=kernel_size, pool_size=(pool_size, pool_size), num_epochs=epochs,
    layer2_count=layer2_count, layer3_count=layer3_count,
    layer1_count=layer1_count, batch_size=i, show_graph=False)
    b_data.append((i, history.history['acc']))

# Pool Size
p = [x for x in range(2, 16)] # 2, 15
p_data = []
for i in p:
    _,history = cnn.cnn_train(train_size=train_size, test_size=test_size,
    kernel_size=kernel_size, pool_size=(i, i), num_epochs=epochs,
    layer2_count=layer2_count, layer3_count=layer3_count,
    layer1_count=layer1_count, batch_size=batch_size, show_graph=False)
    p_data.append((i, history.history['acc']))


plot_graph(e_data, 'Varying number of Epochs', 'Epochs', 'Accuracy')
plot_graph(l1_data, 'Varying Layer 1 Neuron Count', 'Number of Neurons', 'Accuracy')
plot_graph(l2_data, 'Varying Layer 2 of Neuron Count', 'Number of Neurons', 'Accuracy')
plot_graph(l3_data, 'Varying Layer 3 of Neuron Count', 'Number of Neurons', 'Accuracy')
plot_graph(k_data, 'Varying Kernel Size', 'Kernel Size (x * x)', 'Accuracy')
plot_graph(b_data, 'Varying Batch Size', 'Batch Size', 'Accuracy')
plot_graph(p_data, 'Varying Pool Size', 'Pool Size (x * x)', 'Accuracy')