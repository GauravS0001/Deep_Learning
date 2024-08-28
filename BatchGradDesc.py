import random
import math

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = random.uniform(-1, 1)
        self.learning_rate = learning_rate

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
            total_output_error = 0
            total_hidden_error = [0] * len(self.weights_hidden_output)
            
            for inputs, target in zip(X, y):
                hidden_layer_output, predicted_output = self.forward(inputs)

                error_output = target - predicted_output
                delta_output = error_output * self.sigmoid_derivative(predicted_output)

                delta_hidden_layer = [
                    delta_output * w * self.sigmoid_derivative(hidden_out)
                    for w, hidden_out in zip(self.weights_hidden_output, hidden_layer_output)
                ]

                total_output_error += delta_output
                total_hidden_error = [
                    total_hidden_error[i] + delta_hidden_layer[i]
                    for i in range(len(delta_hidden_layer))
                ]

            # Update weights after processing entire dataset
            self.weights_hidden_output = [
                w + self.learning_rate * total_output_error * hidden_out
                for w, hidden_out in zip(self.weights_hidden_output, hidden_layer_output)
            ]
            self.bias_output += self.learning_rate * total_output_error

            for i in range(len(self.weights_input_hidden)):
                self.weights_input_hidden[i] = [
                    w + self.learning_rate * total_hidden_error[j] * inp
                    for j, (w, inp) in enumerate(zip(self.weights_input_hidden[i], inputs))
                ]
            self.bias_hidden = [
                b + self.learning_rate * total_hidden_error[i]
                for i, b in enumerate(self.bias_hidden)
            ]

    def predict(self, inputs):
        _, output = self.forward(inputs)
        return output