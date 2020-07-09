import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# importing the dataset fashion MNIST
fashion_mnist_dataset = keras.datasets.fashion_mnist

# Loading traning and testing set from fashion MNINST dataset.
# The function load_data already separate the dataset and two groups called train and test.
# This groups will be used to train and test my machine learn model after your implementation
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_dataset.load_data()

# Each imagem in the Fashion MNIST dataset is mapped to a single label. This label is a value between 0 and 9.
# Bellow i'm declaring the name of each class present in this dataset following the sequence of the dataset wher the
# label 0 represent a T-shirt/top and the label 9 represent a Ankle boot
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"The training set has {train_images.shape[0]} images with {train_images.shape[1]} x {train_images.shape[2]} pixels")
print(f"The testing set has {test_images.shape[0]} images with {test_images.shape[1]} x {test_images.shape[2]} pixels")


show_image = False
while show_image:
    image_number = input("Type the number of imagem to be shown (type \'q\' to jump this process):")
    if image_number.isnumeric():
        number = int(image_number)
        plt.figure()
        plt.imshow(train_images[number])
        plt.colorbar()
        plt.grid(False)
        plt.show()
    else:
        if image_number is 'q':
            break
        else:
            print("type a valid value!")

# Before to start the trainig process will be necessary to scale the range of the images. All images of the dataset must be scaled in the range
# of 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Now i'll create mt neural network model that will be trained to learn the paterns of those images.
# This model will be a sequencial neural network with one hidden layer.
# The first layer of the of the model will be the Flatten layer. This layer will be resposible to reformat the data of the dataset.
# Each imagem in MNIST dataset is a imagem with 28 x 28 pixels that will be converted by flatten layer to an array with 28x28=784 values.
# It only will reformat the dataset

# The second layer of the model will be a Dense Layer. The Dense layer is a densely connected or fully connected neural layer where each output of the flatten layer
# is connected with all elements of the second layer of the model. The second layer will use an activation function called ReLU and has 128 nodes (or neurons).

# The third layer of the model can be called as output layer and it is responsible to return show the result of the classification process did by the model.
# This layer has 10 nodes wher each node will return the probability of each imagem is in the classe represented by this node.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])


# Before create the model i need to configure my module adding few settings.
# There settings are added in the moddule during the compile step
# The first setting added in the model is the optimizer method. For this example i'm using the algorithm Adam.
# Adam is a optimizarion algorithm that implement a stochastic gradient descent method. This method will be used
# to define how the model will be updated based on the data and loss function

# The second setting is the loss function. This measures how accurate the model is during training. For this example
# i'm using the SparseCategoricalCrossentropy that measures the performance of a classification model whose output is
# a probability value between 0 and 1

# The third setting is the metric used to monitor the training and testing steps.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# The next step is train my model using the function fit in the model and passing as parameter the training dataset and
# the number of the epochs that will be used to train the model
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = model.predict(test_images)


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


