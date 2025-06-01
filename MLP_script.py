import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_directory(directory_path):
    """
    Create and change into the specified directory.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' is ready.")
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")
    os.chdir(directory_path)
    print("Current Directory:", os.getcwd())


class Transformation:
    """Activation functions and their derivatives."""

    @staticmethod
    def relu(Z):
        return np.maximum(Z, 0)

    @staticmethod
    def relu_deriv(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        ## subtract max for numerical stability.
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    @staticmethod
    def softmax_deriv(Z, epsilon=2e-10):
        ## numerical derivative (illustrative only)
        return (Transformation.softmax(Z + epsilon) + Transformation.softmax(Z)) / epsilon


class Layer:
    """
    A neural network layer.
    
    For the input layer, no weights are stored.
    For subsequent layers, weights (W) and biases (b) are assigned during network construction.
    """

    def __init__(self, size, activation=None):
        self.size = size
        self.W = None
        self.b = None

        if activation == 'relu':
            self.activation = Transformation.relu
            self.activation_deriv = Transformation.relu_deriv
        elif activation == 'softmax':
            self.activation = Transformation.softmax
            self.activation_deriv = None  ## special handling in backprop.
        else:
            self.activation = None
            self.activation_deriv = None

    def __add__(self, other):
        """
        Allows chaining layers using the '+' operator.
        """
        if isinstance(other, Layer):
            return Network(self, other)
        elif isinstance(other, Network):
            other.connect(self, other.layers[0])
            other.layers.insert(0, self)
            return other
        else:
            raise TypeError("Can only add a Layer or Network instance.")

    def get_W_b(self):
        return self.W, self.b

    def get_Z_A(self, layer_input):
        if self.W is not None:
            ## using row-oriented multiplication: (num_samples, features) dot (features, neurons)
            Z = np.dot(layer_input, self.W) + self.b
            A = self.activation(Z) if self.activation is not None else Z
            return Z, A
        else:
            return None, layer_input


class InitializedNetwork:
    """
    Initialize a base network by connecting layers.
    """

    def __init__(self, *layers):
        self.layers = []
        for layer in layers:
            self.add_layer(layer)

    def connect(self, previous_layer, current_layer):
        """
        Connect two layers by initializing weights and biases in the receiving layer.
        The weight matrix has shape (previous_layer.size, current_layer.size).
        """

        input_features_size = previous_layer.size
        output_features_size = current_layer.size
        current_layer.W = np.random.rand(input_features_size, output_features_size) - 0.5
        current_layer.b = np.random.rand(output_features_size) - 0.5
                

    def add_layer(self, layer):
        if self.layers:
            self.connect(self.layers[-1], layer)
        self.layers.append(layer)

    def add_network(self, network):
        if self.layers and network.layers:
            self.connect(self.layers[-1], network.layers[0])
        self.layers.extend(network.layers)

    def __add__(self, other):
        if isinstance(other, Layer):
            self.add_layer(other)
        elif isinstance(other, Network):
            self.add_network(other)
        else:
            raise TypeError("Can only add a Layer or Network instance.")
        return self

    def __getitem__(self, index):
        return self.layers[index]

    def __setitem__(self, index, layer):
        self.layers[index] = layer

    def __delitem__(self, index):
        if 0 < index < len(self.layers) - 1:
            self.connect(self.layers[index - 1], self.layers[index + 1])
        del self.layers[index]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)


class PropagatingNetwork(InitializedNetwork):
    """
    Add gradient_descent to the InitializedNetwork.
    """

    def forward_prop(self, X):
        layers_Z = []
        layers_A = []
        A = X
        for layer in self.layers:
            Z, A = layer.get_Z_A(A)
            layers_Z.append(Z)
            layers_A.append(A)
        return layers_Z, layers_A

    @staticmethod
    def one_hot(Y):
        ## create one-hot encoding with shape (num_samples, num_classes)
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y

    def get_first_dZ(self, Y_pred, Y):
        one_hot_Y = self.one_hot(Y)
        return Y_pred - one_hot_Y

    def get_next_dZ(self, next_W, dZ_next, activation_deriv, Z):
        ## propagate error backwards: (dZ_next dot next_W.T) * f'(Z)
        return np.dot(dZ_next, next_W.T) * activation_deriv(Z)

    def get_layers_dZ(self, layers_Z, Y_pred, Y):
        """
        Compute gradients (dZ) for each weighted layer.
        (Weighted layers are those from index 1 onward.)
        """
        weighted_layers = len(self.layers) - 1  # Exclude input layer.
        dZ_list = [None] * weighted_layers
        ## output layer:
        dZ_list[-1] = self.get_first_dZ(Y_pred, Y)
        ## hidden layers:
        for i in range(weighted_layers - 2, -1, -1):
            dZ_list[i] = self.get_next_dZ(
                self.layers[i + 2].W,  # Weights of the next layer.
                dZ_list[i + 1],
                self.layers[i + 1].activation_deriv,
                layers_Z[i + 1]
            )
        return dZ_list

    def get_layers_inputs(self, X, layers_Z, layers_A):
        """
        For each weighted layer (index 1 to end), the input is the activation from the previous layer.
        """
        inputs = []
        for i in range(1, len(self.layers)):
            inputs.append(layers_A[i - 1])
        return inputs

    @staticmethod
    def get_dW_db(dZ, layer_input, m):
        ## m being the batch size
        ## layer_input shape: (m, input_features)
        ## dZ shape: (m, output_features)
        dW = (1 / m) * np.dot(layer_input.T, dZ)
        db = (1 / m) * np.sum(dZ, axis=0)
        return dW, db

    def get_gradients(self, i, dZ_list, layers_input, m):
        dZ = dZ_list[i]
        layer_input = layers_input[i]
        return self.get_dW_db(dZ, layer_input, m)

    def update_parameters(self, layers_Z, layers_A, X, Y, alpha=0.1):
        m = X.shape[0]
        Y_pred = layers_A[-1]
        dZ_list = self.get_layers_dZ(layers_Z, Y_pred, Y)
        layers_input = self.get_layers_inputs(X, layers_Z, layers_A)
        ## update parameters for each weighted layer (layers 1, 2, ...).
        for i in range(len(dZ_list)):
            dW, db = self.get_gradients(i, dZ_list, layers_input, m)
            self.layers[i + 1].W -= alpha * dW
            self.layers[i + 1].b -= alpha * db

    def gradient_descent(self, X, Y, alpha=0.1, iterations=100):
        """
        Runs gradient descent for a given number of iterations.
        """
        layers_Z, layers_A = self.forward_prop(X)
        self.update_parameters(layers_Z, layers_A, X, Y, alpha)
        return layers_Z, layers_A
    

class Network(PropagatingNetwork):
    """
    Add user functionality to the PropagatingNetwork such as training and saving.
    """
    def __init__(self, *layers):
        super().__init__(*layers)

    def __rrshift__(self, data):
        """
        >> operator: data >> self
          - If data is (X, Y) train the network using TrainingChain.
          - If data is (X), return Y.
        """
        if not isinstance(data, (list, tuple)):
            raise ValueError("Data needs to be a list or a tuple.")
        elif len(data) not in (1, 2):
            raise ValueError("Data needs to have a length of 1 for (X) or 2 for (X, Y)")
        
        if len(data) == 2:
            return TrainingChain(self, data)
        elif len(data) == 1:
            X = data[0]
            _, layers_A = self.forward_prop(X)
            return layers_A[-1]
    
    def load(self):
        """
        Load weights and biases from .npy files.
        """
        for i in range(1, len(self.layers)):
            self.layers[i].W = np.load(f'W{i}.npy')
            self.layers[i].b = np.load(f'b{i}.npy')

    def save(self):
        """
        Save weights and biases to .npy files.
        """
        for i in range(1, len(self.layers)):
            np.save(f'W{i}.npy', self.layers[i].W)
            np.save(f'b{i}.npy', self.layers[i].b)

    @staticmethod
    def get_predictions(A):
        return np.argmax(A, axis=1)

    def get_accuracy(self, predictions, Y):
        accuracy = np.sum(predictions == Y) / Y.size
        return accuracy


class TrainingChain:
    """
    Helper class returned by the >> operator.
    It stores the model, training data, and accumulates an internal product of iterations.
    When the TrainingChain is garbage-collected, it automatically triggers training.
    """
    def __init__(self, model, training_data):
        self.model = model
        self.data = training_data  ## tuple (X, Y)
        self.total_iterations = 1  ## start with a neutral multiplier.
        self.triggered = False

    def __mul__(self, factor):
        if not isinstance(factor, int):
            raise TypeError("Multiplication factor must be an integer.")
        self.total_iterations *= factor
        return self
        
    def _run(self, alpha=0.1):
        if not self.triggered:
            X, Y = self.data
            #m = X.shape[0]  ## not used here
            for i in range(1, self.total_iterations + 1):
                layers_Z, layers_A = self.model.gradient_descent(X, Y, alpha=alpha)
                if i % 10 == 0:
                    print(f"Iteration: {i}")
                    preds = self.model.get_predictions(layers_A[-1])
                    accuracy = self.model.get_accuracy(preds, Y)
                    print(f'Accuracy: {accuracy:.3f}')
            self.triggered = True
        return self.model

    def __del__(self):
        if not self.triggered:
            self._run()


def plot_number(pixels_array):
    image_2d = np.squeeze(pixels_array).reshape(28, 28)

    fig, ax = plt.subplots()
    ax.imshow(image_2d, cmap='gray')
    ax.axis('off')
    fig.canvas.manager.set_window_title("Press anything to continue")

    plt.draw()

    plt.waitforbuttonpress()
    plt.close(fig)


def main():
    train = False  ## set to False for testing mode.
    data = pd.read_csv('train.csv').to_numpy()
    split_idx = int(0.8 * len(data))
    data_train = data[:split_idx]
    data_test = data[split_idx:]
    
    if train:
        print("TRAINING MODE")
        ## in this case, first column contains labels and the rest pixel values.
        Y_train = data_train[:, 0]
        X_train = data_train[:, 1:] / 255.0
        ## Define layers. The input layer has no weights.
        L0 = Layer(784)
        L1 = Layer(100, 'relu')
        L2 = Layer(50, 'relu')
        L3 = Layer(10, 'softmax')

        model = L0 + L1 + L2 + L3
        ## to load pre-trained weights before training, uncomment the load:
        ## model.load()

        ## chaining multiplications automatically triggers training via __del__.
        ## the expression below computes 50 * 3 * 2 = 300 iterations.
        ((((X_train, Y_train) >> model) * 50) * 3 ) * 2
        model.save()
    else:
        print("TESTING MODE")
        ## np.random.shuffle(data_test)
        Y_test = data_test[:, 0]
        X_test = data_test[:, 1:] / 255.0

        L0 = Layer(784)
        L1 = Layer(100)
        L2 = Layer(50, 'relu')
        L3 = Layer(10, 'softmax')
        model = L0 + L1 + L2 + L3
        model.load()
        
        probs = (X_test,) >> model
        preds = model.get_predictions(probs)
        accuracy = model.get_accuracy(preds, Y_test)
        print(f"Test Accuracy: {accuracy:.3f}")
        
        
        while True:
            i = int(input("Enter sample index (-1 to exit): "))
            if i == -1:
                break
            
            X = X_test[i:i+1, :]
            Y = Y_test[i:i+1]
            probs = (X,) >> model
            preds = Network.get_predictions(probs)
            print(f"model prediction: {preds}")
            print(f"ground truth {Y}")
            plot_number(X)


if __name__ == '__main__':
    main()