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

    def forward(self, input: np.ndarray[any, np.dtype[np.float64]]) -> None:
        self.output = np.dot(input, self.weights) + self.biases;
    
    def show_result(self) -> None:
        print(self.output);

    def to_string(self) -> None:
        print('weights', self.weights);
        print('biases', self.biases);  
        print('output', self.output);      

class Activation_Relu:
    output : np.ndarray[np.float64] = None;

    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray[any, np.dtype[np.float64]]):
        self.output = np.maximum(0, inputs);

    def show_result(self) -> None:
            print(self.output);

    def show_result(self) -> None:
        print(self.output);


activation = Activation_Relu();

layer1 = Layer_Dense(2, 5);
layer1.forward(X);

activation.forward(layer1.output);

activation.show_result();
 