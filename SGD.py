import numpy as np
import tensorflow as tf
import random

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data("archive")

# Normalize the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the cost function
def cost_function(vector, example):
    x, y = example
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y_pred = tf.nn.softmax(tf.matmul(x, vector))
    return -tf.reduce_sum(y * tf.math.log(y_pred))

# Define the gradient of the cost function
def gradient(vector, example):
    x, y = example
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    y_pred = tf.nn.softmax(tf.matmul(x, vector))
    return tf.matmul(tf.transpose(x), (y_pred - y))

# Initialize the parameters
start = np.random.randn(784, 10)

# Shuffle data and labels
# Shuffle data and labels
def shuffle(data):
    """
    Shuffle the data and labels in place.

    Parameters:
    data (tuple): The data and labels to shuffle.

    Returns:
    None
    """
    data_list = list(data)
    n = len(data_list[0])
    indices = list(range(n))
    random.shuffle(indices)
    data_list[0] = data_list[0][indices]
    data_list[1] = data_list[1][indices]
    return tuple(data_list)

# Run the stochastic gradient descent algorithm
def stochastic_gradient_descent(
    cost_function, gradient, data, start, learn_rate, n_iter=50, tolerance=1e-6
):
    """
    Stochastic gradient descent algorithm.

    Parameters:
    cost_function (callable): The cost function to minimize.
    gradient (callable): The gradient of the cost function.
    data (tuple): The training data and labels.
    start (array-like): The starting parameters.
    learn_rate (float): The learning rate.
    n_iter (int): The maximum number of iterations.
    tolerance (float): The tolerance for convergence.

    Returns:
    array-like: The final parameters.
    """

    # Initialize the parameters
    vector = np.array(start)

    # Iterate over the data
    for _ in range(n_iter):

        # Shuffle the data
        shuffle(data)

        # Iterate over the examples
        for i in range(len(data[0])):

            # Calculate the gradient of the cost function
            grad = gradient(vector, (data[0][i].reshape(-1, 784), data[1][i]))


            # Update the parameters
            vector -= learn_rate * grad

            # Check for convergence
            if np.linalg.norm(grad) < tolerance:
                break

    return vector

result = stochastic_gradient_descent(
    cost_function, gradient, (x_train, y_train), start, learn_rate=0.01, n_iter=50
)

# Evaluate the model on the test dataset
x_test_flat = x_test.reshape(-1, 784)
y_pred = np.argmax(tf.nn.softmax(tf.matmul(tf.convert_to_tensor(x_test_flat, dtype=tf.float64), result)), axis=1)
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print("Test accuracy:", accuracy)

 
 # visualization 
