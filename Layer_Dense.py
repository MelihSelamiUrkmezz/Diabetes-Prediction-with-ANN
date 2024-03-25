import numpy as np;
from ActivationFunction import Sigmoid;
class Layer_Dense:
    weights: np.ndarray[any, np.dtype[np.float64]] = None;
    weights2: np.ndarray[any, np.dtype[np.float64]] = None;
    biases : np.ndarray[np.float64] = None;
    output: np.ndarray[any, np.dtype[np.float64]] = None;
    sigmoid = Sigmoid();
    np.random.seed(1)
    def __init__(self, n_inputs: int = 3, n_neurons: int = 4) -> None:
        self.weights = 2 * np.random.random((n_inputs, n_neurons)) - 1; # not necessary Tranpos
        self.weights2 = 2 * np.random.random((n_neurons, 1)) - 1;
        self.biases = np.zeros((1, n_neurons));

    def forward(self, input: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        self.output = np.dot(input, self.weights) + self.biases;
        return self.output;

    def predict(self, input: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        # for now we use 1 hidden layer with 3 neurons
        layer_1 = self.sigmoid.forward(np.dot(input, self.weights));
        layer_2 = self.sigmoid.forward(np.dot(layer_1, self.weights2));

        return layer_2;

    def train(self, input_training: np.ndarray, output_training: np.ndarray, apochs: int):

        for apoch in range(apochs):

            layer_input = input_training;
            layer_1 = self.sigmoid.forward(np.dot(layer_input, self.weights));
            layer_2 = self.sigmoid.forward(np.dot(layer_1, self.weights2));
        
            layer_2_error = output_training - layer_2;
            delta_2 = layer_2_error * self.sigmoid.derivative(layer_2);

            if apoch % 500 == 0:
                print('Final error after', apoch, 'iterations =', np.mean(np.abs(layer_2_error)));
    

            layer_1_error = np.dot(delta_2, self.weights2.T);
            delta_1 = layer_1_error * self.sigmoid.derivative(layer_1);
    
            self.weights2 += np.dot(layer_1.T, delta_2);
            self.weights += np.dot(layer_input.T, delta_1);
      
    
    def show_result(self) -> None:
        print(self.output);

    def to_string(self) -> None:
        print('weights', self.weights);
        print('biases', self.biases);  
        print('output', self.output);      