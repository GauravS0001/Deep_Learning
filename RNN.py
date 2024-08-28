import math

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        
        self.input_to_hidden_weights = [[0.01 for _ in range(hidden_size)] for _ in range(input_size)]
        self.hidden_to_hidden_weights = [[0.01 for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.hidden_to_output_weights = [[0.01 for _ in range(output_size)] for _ in range(hidden_size)]
        
        self.hidden_biases = [0.01 for _ in range(hidden_size)]
        self.output_biases = [0.01 for _ in range(output_size)]
        
        self.learning_rate = learning_rate
        
        self.hidden_state = None
        self.previous_hidden_state = None
        
        # For storing gradients
        self.d_input_to_hidden_weights = None
        self.d_hidden_to_hidden_weights = None
        self.d_hidden_to_output_weights = None
        self.d_hidden_biases = None
        self.d_output_biases = None

    def activation_function(self, x):
        # Activation fun. tanh 
        return [(2 / (1 + math.exp(-2 * xi))) - 1 for xi in x]

    def activation_function_derivative(self, x):
        # Derivative of tanh
        return [1 - xi**2 for xi in x]
    
    def forward(self, input_sequence):
        self.hidden_states = []
        self.hidden_state = [0 for _ in range(len(self.hidden_to_hidden_weights))]
        
      
        for input_vector in input_sequence:
          
            new_hidden_state = []
            for hidden_index in range(len(self.hidden_state)):
                input_sum = sum(input_vector[input_index] * self.input_to_hidden_weights[input_index][hidden_index] for input_index in range(len(input_vector)))
                hidden_sum = sum(self.hidden_state[previous_hidden_index] * self.hidden_to_hidden_weights[previous_hidden_index][hidden_index] for previous_hidden_index in range(len(self.hidden_state)))
                new_hidden_value = input_sum + hidden_sum + self.hidden_biases[hidden_index]
                new_hidden_state.append(new_hidden_value)
            
            # Update hidden state with activation function
            self.hidden_state = self.activation_function(new_hidden_state)
            
            # Store the hidden state
            self.hidden_states.append(self.hidden_state)
       

        output_vector = [0 for _ in range(len(self.output_biases))]
        for output_index in range(len(output_vector)):
            output_sum = sum(self.hidden_state[hidden_index] * self.hidden_to_output_weights[hidden_index][output_index] for hidden_index in range(len(self.hidden_state)))
            output_vector[output_index] = output_sum + self.output_biases[output_index]
        
        return output_vector

    def compute_loss(self, output, target):
        # MSE
        return sum((o - t) ** 2 for o, t in zip(output, target)) / len(target)
    
    def backward(self, input_sequence, target_sequence):
        # Initialize gradients
        self.d_input_to_hidden_weights = [[0 for _ in range(len(self.hidden_state))] for _ in range(len(input_sequence[0]))]
        self.d_hidden_to_hidden_weights = [[0 for _ in range(len(self.hidden_state))] for _ in range(len(self.hidden_state))]
        self.d_hidden_to_output_weights = [[0 for _ in range(len(self.hidden_state))] for _ in range(len(self.hidden_state))]
        self.d_hidden_biases = [0 for _ in range(len(self.hidden_state))]
        self.d_output_biases = [0 for _ in range(len(self.output_biases))]

        
        self.hidden_states = []
        self.forward(input_sequence)

        # Compute gradients BPTT
        output_gradients = [2 * (self.hidden_state[i] - target_sequence[i]) for i in range(len(target_sequence))]
        
        # Compute gradients for hidden-to-output weights
        for output_index in range(len(self.output_biases)):
            for hidden_index in range(len(self.hidden_state)):
                self.d_hidden_to_output_weights[hidden_index][output_index] = output_gradients[output_index] * self.hidden_state[hidden_index]
                self.d_output_biases[output_index] = output_gradients[output_index]
        
        # Backpropagate 
        for t in reversed(range(len(input_sequence))):
            input_vector = input_sequence[t]
            current_hidden_state = self.hidden_states[t]
            prev_hidden_state = self.hidden_states[t - 1] if t > 0 else [0 for _ in range(len(self.hidden_state))]

            # Compute the gradient for the hidden-to-hidden weights
            for hidden_index in range(len(self.hidden_state)):
                for prev_hidden_index in range(len(self.hidden_state)):
                    self.d_hidden_to_hidden_weights[prev_hidden_index][hidden_index] += output_gradients[hidden_index] * prev_hidden_state[prev_hidden_index]

            # Compute the gradient for the input-to-hidden weights
            for input_index in range(len(input_vector)):
                for hidden_index in range(len(self.hidden_state)):
                    self.d_input_to_hidden_weights[input_index][hidden_index] += output_gradients[hidden_index] * input_vector[input_index]

            # Compute the gradient for hidden biases
            self.d_hidden_biases = [output_gradients[hidden_index] for hidden_index in range(len(self.hidden_state))]

        # Update weights and biases
        self.update_weights()

    def update_weights(self):
        # Update weights and biases using gradients and learning rate
        for i in range(len(self.input_to_hidden_weights)):
            for j in range(len(self.input_to_hidden_weights[0])):
                self.input_to_hidden_weights[i][j] -= self.learning_rate * self.d_input_to_hidden_weights[i][j]

        for i in range(len(self.hidden_to_hidden_weights)):
            for j in range(len(self.hidden_to_hidden_weights[0])):
                self.hidden_to_hidden_weights[i][j] -= self.learning_rate * self.d_hidden_to_hidden_weights[i][j]

        for i in range(len(self.hidden_to_output_weights)):
            for j in range(len(self.hidden_to_output_weights[0])):
                self.hidden_to_output_weights[i][j] -= self.learning_rate * self.d_hidden_to_output_weights[i][j]

        for i in range(len(self.hidden_biases)):
            self.hidden_biases[i] -= self.learning_rate * self.d_hidden_biases[i]

        for i in range(len(self.output_biases)):
            self.output_biases[i] -= self.learning_rate * self.d_output_biases[i]


input_size = 3
hidden_size = 5
output_size = 2

rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate=0.01)


input_sequence = [
    [1, 0, -1],
    [0, 1, 1],
    [-1, -1, 0]
]

target_sequence = [1, 0]  

# Forward pass
output = rnn.forward(input_sequence)
print("Output of the RNN:", output)

# Backward pass to update weights
rnn.backward(input_sequence, target_sequence)