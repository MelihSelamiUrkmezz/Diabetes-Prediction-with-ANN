import numpy as np;
import nnfs;
from nnfs.datasets import spiral_data; # or you can use ./DummyDataGenerator for this. just uncomment underline :)
# from DummyDataGenerator import spiral_data as mySprialData

nnfs.init();

X, y = spiral_data(100, 3);
class Layer_Dense:
    weights: np.ndarray[any, np.dtype[np.float64]] = None;
    biases : np.ndarray[np.float64] = None;
    output: np.ndarray[any, np.dtype[np.float64]] = None;

    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons); # not necessary Tranpos
        self.biases = np.zeros((1, n_neurons));

    def forward(self, input: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        self.output = np.dot(input, self.weights) + self.biases;
        return self.output;
    
    def show_result(self) -> None:
        print(self.output);

    def to_string(self) -> None:
        print('weights', self.weights);
        print('biases', self.biases);  
        print('output', self.output);      

class Activation_Relu:
    output : np.ndarray[any, np.dtype[np.float64]] = None;

    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        self.output = np.maximum(0, inputs);
        return self.output;

    def show_result(self) -> None:
        print(self.output);


class Activation_Softmax: # exponential function with normalization
    output : np.ndarray[any, np.dtype[np.float64]] = None;
    
    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)); # for escape infinite value otherwiise code throw exception (overflow prevention)
        normalize_exp_values = exp_values / np.sum(exp_values, axis=1, keepdims=True);
        self.output = normalize_exp_values;
        return self.output;

    def show_result(self) -> None:
        print(self.output);


activation_relu = Activation_Relu();
activation_softmax = Activation_Softmax();

layer1 = Layer_Dense(2, 5);
layer1.forward(X);
activation_relu.forward(layer1.output);

layer2 = Layer_Dense(5, 2);
layer2.forward(activation_relu.output);
activation_softmax.forward(layer2.output);

activation_softmax.show_result();
 