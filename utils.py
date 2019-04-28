from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def read_image(filename, img_shape):
    img = Image.open(filename)
    img = img.convert(mode='L')  # Convert to grayscale
    img = img.resize(img_shape)

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
