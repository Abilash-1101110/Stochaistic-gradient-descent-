# Stochaistic-gradient-descent-
The provided code implements a stochastic gradient descent (SGD) algorithm for training a neural network on the MNIST dataset. Let's break down the mathematical components of the code:

1. **Cost Function**:
   The cost function used is the cross-entropy loss. It measures the difference between the predicted probabilities (after softmax) and the actual labels.

   Mathematically, for a single example:
   \[ \text{Cost} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) \]
   Where:
   - \(C\) is the number of classes (in this case, 10 for MNIST).
   - \(y_i\) is the actual label (one-hot encoded).
   - \(\hat{y}_i\) is the predicted probability for class \(i\) after softmax.

2. **Gradient of the Cost Function**:
   The gradient of the cost function with respect to the parameters (weights) is computed. This gradient is used to update the parameters during optimization (SGD).

   Mathematically, for a single example:
   \[ \nabla_{\theta} \text{Cost} = X^T (\hat{y} - y) \]
   Where:
   - \(X\) is the input features (reshaped to a column vector).
   - \(y\) is the actual label (one-hot encoded).
   - \(\hat{y}\) is the predicted probability after softmax.

3. **Stochastic Gradient Descent (SGD)**:
   The SGD algorithm is used to minimize the cost function by updating the parameters (weights) in the opposite direction of the gradient.

   Mathematically, the update step for parameter \(w\) using the gradient \(g\) and learning rate \(\alpha\):
   \[ w_{\text{new}} = w_{\text{old}} - \alpha \cdot g \]

   The parameters are updated iteratively using mini-batches of training data.

4. **Shuffling Data**:
   The training data and labels are shuffled to introduce randomness and avoid overfitting to the order of data samples during training.

5. **Evaluation Metrics**:
   The accuracy of the model is evaluated on the test dataset after training. It measures the proportion of correctly classified samples.

In summary, the provided code implements a simple neural network training pipeline using SGD with cross-entropy loss for classification on the MNIST dataset.
