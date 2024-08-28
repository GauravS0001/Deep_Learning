'''
A Perceptron is the simplest type of artificial neural network and serves as the basic building block for more complex neural networks. It's a binary classifier that maps its input to an output decision using a linear function.
Structure of a Perceptron

A perceptron consists of:

    Input Values (Features): These are the attributes of the data (e.g., x1,x2,…,xnx1​,x2​,…,xn​).
    Weights: Each input has an associated weight (w1,w2,…,wnw1​,w2​,…,wn​), which determines the input's importance.
    Bias: An additional parameter (bb) added to the weighted sum, helping the model adjust its decision boundary.
    Activation Function: A function that determines the output based on the weighted sum of inputs plus bias.

Perceptron Equation

The perceptron computes a weighted sum of the input values and then applies the activation function. The equation for a perceptron is:
z=∑i=1nwi⋅xi+b
z=i=1∑n​wi​⋅xi​+b

Where:

    xixi​ are the input features.
    wiwi​ are the weights.
    bb is the bias.
    zz is the weighted sum of inputs and bias.

The activation function then determines the output (yy):
y={1if z≥00if z<0
y={10​if z≥0if z<0​
Working of a Perceptron

    Initialization: Start with random weights and bias.
    Weighted Sum Calculation: Compute the weighted sum of inputs plus bias.
    Activation Function: Apply the step function to determine the output.
    Learning: Adjust weights and bias based on the error (difference between predicted output and actual output).
    Iteration: Repeat the process until the model correctly classifies the data or reaches a maximum number of iterations.
'''


import numpy as np

# Step function as the activation function
def step_function(z):
    return 1 if z >= 0 else 0


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return step_function(z)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  
y = np.array([0, 1, 1, 1])  


perceptron = Perceptron(input_size=2)
perceptron.train(X, y, epochs=10)


for xi in X:
    print(f"Input: {xi}, Prediction: {perceptron.predict(xi)}")