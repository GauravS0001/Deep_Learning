import random
import math

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
       
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = random.uniform(-1, 1)
        self.learning_rate = learning_rate,

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
       
        hidden_layer_input = []
        for i in range(len(self.weights_input_hidden[0])):
            sum_input = sum(w * inp for w, inp in zip(self.weights_input_hidden, inputs))
            hidden_layer_input.append(self.sigmoid(sum_input + self.bias_hidden[i]))

     
        output = sum(w * hidden_out for w, hidden_out in zip(self.weights_hidden_output, hidden_layer_input))
        output = self.sigmoid(output + self.bias_output)
        return hidden_layer_input, output

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                # Forward pass
                hidden_layer_output, predicted_output = self.forward(inputs)

                
                error_output = target - predicted_output

                # Backpropagation
                delta_output = error_output * self.sigmoid_derivative(predicted_output)
                delta_hidden_layer = [
                    delta_output * w * self.sigmoid_derivative(hidden_out)
                    for w, hidden_out in zip(self.weights_hidden_output, hidden_layer_output)
                ]

                # Update weights and biases
                self.weights_hidden_output = [
                    w + self.learning_rate * delta_output * hidden_out
                    for w, hidden_out in zip(self.weights_hidden_output, hidden_layer_output)
                ]
                self.bias_output += self.learning_rate * delta_output

                for i in range(len(self.weights_input_hidden)):
                    self.weights_input_hidden[i] = [
                        w + self.learning_rate * delta_hidden * inp
                        for w, inp, delta_hidden in zip(self.weights_input_hidden[i], inputs, delta_hidden_layer)
                    ]
                self.bias_hidden = [
                    b + self.learning_rate * delta_hidden
                    for b, delta_hidden in zip(self.bias_hidden, delta_hidden_layer)
                ]

    def predict(self, inputs):
        _, output = self.forward(inputs)
        return output


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]


nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)

nn.train(X, y, epochs=10000)

for inputs in X:
    print(f"Input: {inputs}, Predicted Output: {nn.predict(inputs):.4f}")
