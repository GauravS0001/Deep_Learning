import math
import random

class SimpleLSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W_f = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(input_size + hidden_size)]
        self.W_i = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(input_size + hidden_size)]
        self.W_c = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(input_size + hidden_size)]
        self.W_o = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(input_size + hidden_size)]
        
        self.b_f = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        self.b_i = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        self.b_c = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        self.b_o = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
        
        self.W_y = [[random.uniform(-0.1, 0.1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b_y = [random.uniform(-0.1, 0.1) for _ in range(output_size)]
        
    def sigmoid(self, x):
        return [1 / (1 + math.exp(-xi)) for xi in x]

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return [si * (1 - si) for si in sig]

    def tanh(self, x):
        return [(2 / (1 + math.exp(-2 * xi))) - 1 for xi in x]

    def tanh_derivative(self, x):
        tanh_x = self.tanh(x)
        return [1 - tx**2 for tx in tanh_x]

    def forward(self, inputs):
        self.hidden_states = []
        self.cell_states = []
        self.outputs = []
        
        h_t = [0] * self.hidden_size  # Initial hidden state
        c_t = [0] * self.hidden_size  # Initial cell state
        
        for x_t in inputs:
            xh_t = x_t + h_t  # Concatenate input and hidden state
            
            # Forget gate
            f_t = self.sigmoid([sum(w * xh for w, xh in zip(self.W_f[i], xh_t)) + self.b_f[i] for i in range(self.hidden_size)])
            
            # Input gate
            i_t = self.sigmoid([sum(w * xh for w, xh in zip(self.W_i[i], xh_t)) + self.b_i[i] for i in range(self.hidden_size)])
            
            # Candidate cell state
            c_hat_t = self.tanh([sum(w * xh for w, xh in zip(self.W_c[i], xh_t)) + self.b_c[i] for i in range(self.hidden_size)])
            
            # New cell state
            c_t = [f * c + i * c_hat for f, c, i, c_hat in zip(f_t, c_t, i_t, c_hat_t)]
            
            # Output gate
            o_t = self.sigmoid([sum(w * xh for w, xh in zip(self.W_o[i], xh_t)) + self.b_o[i] for i in range(self.hidden_size)])
            
            # New hidden state
            h_t = [o * self.tanh([c])[0] for o, c in zip(o_t, c_t)]
            
            # Compute output
            y_t = [sum(w * h for w, h in zip(self.W_y[i], h_t)) + self.b_y[i] for i in range(self.output_size)]
            
            self.hidden_states.append(h_t)
            self.cell_states.append(c_t)
            self.outputs.append(y_t)
        
        return self.outputs
    
    def compute_loss(self, predicted, target):
        return sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(target)

    def backward(self, inputs, targets):
        # Initialize gradients
        dW_f = [[0] * self.hidden_size for _ in range(self.input_size + self.hidden_size)]
        dW_i = [[0] * self.hidden_size for _ in range(self.input_size + self.hidden_size)]
        dW_c = [[0] * self.hidden_size for _ in range(self.input_size + self.hidden_size)]
        dW_o = [[0] * self.hidden_size for _ in range(self.input_size + self.hidden_size)]
        
        db_f = [0] * self.hidden_size
        db_i = [0] * self.hidden_size
        db_c = [0] * self.hidden_size
        db_o = [0] * self.hidden_size
        
        dW_y = [[0] * self.output_size for _ in range(self.hidden_size)]
        db_y = [0] * self.output_size
        
        dh_next = [0] * self.hidden_size
        dc_next = [0] * self.hidden_size
        
        # Backpropagation
        for t in reversed(range(len(inputs))):
            # Calculate output gradients
            dy = [2 * (self.outputs[t][i] - targets[i]) for i in range(self.output_size)]
            
            # Calculate gradients for W_y and b_y
            for i in range(self.hidden_size):
                for j in range(self.output_size):
                    dW_y[i][j] += dy[j] * self.hidden_states[t][i]
                db_y[j] += dy[j]
            
            # Backpropagate through output gate
            dh = [dy[i] * self.W_y[i][j] + dh_next[i] for i in range(self.hidden_size)]
            
            do = [dh[i] * self.tanh([self.cell_states[t][i]])[0] for i in range(self.hidden_size)]
            dW_o = self.sigmoid_derivative([self.cell_states[t][i]])[0]
            
            # Backpropagate through cell state
            dc = [dc_next[i] + dh[i] * self.hidden_states[t][i] for i in range(self.hidden_size)]
            dc_next = [dc[i] * self.sigmoid_derivative([self.cell_states[t][i]])[0] for i in range(self.hidden_size)]
            
            # Backpropagate through forget gate
            df = [dc[i] * self.cell_states[t - 1][i] if t > 0 else 0 for i in range(self.hidden_size)]
            
            # Backpropagate through input gate
            di = [dc[i] * self.tanh_derivative([self.hidden_states[t][i]])[0] for i in range(self.hidden_size)]
            
            # Backpropagate through candidate cell state
            dc_hat = [di[i] * self.sigmoid([self.cell_states[t][i]])[0] for i in range(self.hidden_size)]
            
            # Compute weight gradients
            for i in range(self.hidden_size):
                for j in range(self.input_size + self.hidden_size):
                    dW_f[j][i] += df[i] * inputs[t][j] if j < self.input_size else dh_next[j - self.input_size]
                    dW_i[j][i] += di[i] * inputs[t][j] if j < self.input_size else dh_next[j - self.input_size]
                    dW_c[j][i] += dc_hat[i] * inputs[t][j] if j < self.input_size else dh_next[j - self.input_size]
                    dW_o[j][i] += do[i] * inputs[t][j] if j < self.input_size else dh_next[j - self.input_size]
                
                db_f[i] += df[i]
                db_i[i] += di[i]
                db_c[i] += dc_hat[i]
                db_o[i] += do[i]
        
        # Update weights and biases
        for i in range(self.input_size + self.hidden_size):
            for j in range(self.hidden_size):
                self.W_f[i][j] -= self.learning_rate * dW_f[i][j]
                self.W_i[i][j] -= self.learning_rate * dW_i[i][j]
                self.W_c[i][j] -= self.learning_rate * dW_c[i][j]
                self.W_o[i][j] -= self.learning_rate * dW_o[i][j]
        
        for i in range(self.hidden_size):
            self.b_f[i] -= self.learning_rate * db_f[i]
            self.b_i[i] -= self.learning_rate * db_i[i]
            self.b_c[i] -= self.learning_rate * db_c[i]
            self.b_o[i] -= self.learning_rate * db_o[i]
        
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.W_y[i][j] -= self.learning_rate * dW_y[i][j]
        
        for i in range(self.output_size):
            self.b_y[i] -= self.learning_rate * db_y[i]


lstm = SimpleLSTM(input_size=3, hidden_size=5, output_size=2)


inputs = [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5]
]

# Targets
targets = [0.4, 0.6]

# Forward pass
outputs = lstm.forward(inputs)
print("Outputs:", outputs)

# Compute loss
loss = lstm.compute_loss(outputs[-1], targets)
print("Loss:", loss)

# Backward pass
lstm.backward(inputs, targets)