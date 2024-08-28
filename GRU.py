import numpy as np

class GRU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W_z = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_z = np.zeros(hidden_size)
        
        self.W_r = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_r = np.zeros(hidden_size)
        
        self.W_h = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_h = np.zeros(hidden_size)
        
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros(output_size)
        
        # Initialize gradients
        self.dW_z = np.zeros_like(self.W_z)
        self.db_z = np.zeros_like(self.b_z)
        
        self.dW_r = np.zeros_like(self.W_r)
        self.db_r = np.zeros_like(self.b_r)
        
        self.dW_h = np.zeros_like(self.W_h)
        self.db_h = np.zeros_like(self.b_h)
        
        self.dW_y = np.zeros_like(self.W_y)
        self.db_y = np.zeros_like(self.b_y)

        self.h_prev = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, X):
        self.X = X
        self.batch_size = X.shape[0]
        self.seq_len = X.shape[1]

        self.h = np.zeros((self.batch_size, self.hidden_size))
        self.h_list = []

        for t in range(self.seq_len):
            x_t = X[:, t, :]

            # Concatenate h_prev and x_t
            concat = np.concatenate((self.h, x_t), axis=1)

            # Update gate
            z_t = self.sigmoid(np.dot(self.W_z, concat.T) + self.b_z.reshape(-1, 1))
            
            # Reset gate
            r_t = self.sigmoid(np.dot(self.W_r, concat.T) + self.b_r.reshape(-1, 1))

            # Candidate hidden state
            concat_r = np.concatenate((r_t * self.h, x_t), axis=1)
            h_tilde = self.tanh(np.dot(self.W_h, concat_r.T) + self.b_h.reshape(-1, 1))

            # Final hidden state
            self.h = (1 - z_t.T) * self.h + z_t.T * h_tilde.T
            self.h_list.append(self.h)

        # Output layer
        self.y = np.dot(self.W_y, self.h.T) + self.b_y.reshape(-1, 1)
        return self.y.T

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_true):
        self.dL_dy = 2 * (self.y.T - y_true) / self.batch_size
        
        self.dW_y = np.dot(self.dL_dy.T, self.h)
        self.db_y = np.sum(self.dL_dy, axis=0)

        # Initialize gradients
        dL_dh = np.dot(self.dL_dy, self.W_y)

        for t in reversed(range(self.seq_len)):
            x_t = self.X[:, t, :]

            h_t = self.h_list[t]
            h_prev = self.h_list[t - 1] if t > 0 else np.zeros_like(h_t)

            z_t = self.sigmoid(np.dot(self.W_z, np.concatenate((h_prev, x_t), axis=1).T) + self.b_z.reshape(-1, 1))
            r_t = self.sigmoid(np.dot(self.W_r, np.concatenate((h_prev, x_t), axis=1).T) + self.b_r.reshape(-1, 1))

            h_tilde = self.tanh(np.dot(self.W_h, np.concatenate((r_t * h_prev, x_t), axis=1).T) + self.b_h.reshape(-1, 1))

            dL_dh_tilde = dL_dh * z_t * (1 - h_tilde ** 2)
            dL_dz = dL_dh * (h_tilde - h_prev) * z_t * (1 - z_t)
            dL_dr = dL_dh * (self.h_list[t - 1] * h_tilde) * r_t * (1 - r_t)

            concat = np.concatenate((self.h_list[t - 1], x_t), axis=1)

            self.dW_h += np.dot(dL_dh_tilde, concat_r.T)
            self.db_h += np.sum(dL_dh_tilde, axis=0)
            self.dW_z += np.dot(dL_dz.T, concat.T)
            self.db_z += np.sum(dL_dz, axis=0)
            self.dW_r += np.dot(dL_dr.T, concat.T)
            self.db_r += np.sum(dL_dr, axis=0)

            dL_dh = np.dot(self.W_h.T, dL_dh_tilde) * r_t

        # Update weights
        self.W_y -= self.learning_rate * self.dW_y
        self.b_y -= self.learning_rate * self.db_y

        self.W_z -= self.learning_rate * self.dW_z
        self.b_z -= self.learning_rate * self.db_z
        
        self.W_r -= self.learning_rate * self.dW_r
        self.b_r -= self.learning_rate * self.db_r

        self.W_h -= self.learning_rate * self.dW_h
        self.b_h -= self.learning_rate * self.db_h


input_size = 3
hidden_size = 4
output_size = 2

gru = GRU(input_size, hidden_size, output_size, learning_rate=0.01)


input_sequence = np.array([
    [[0.5, 0.2, -0.1], [0.1, -0.2, 0.4], [-0.3, 0.1, 0.2]],
    [[0.4, -0.1, 0.3], [0.2, -0.3, 0.5], [-0.1, 0.2, -0.4]]
])

target_sequence = np.array([
    [1, 0],
    [0, 1]
])

# Forward pass
output = gru.forward(input_sequence)
print("Output of the GRU:", output)

# Backward pass to update weights
gru.backward(target_sequence)