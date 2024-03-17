import numpy as np;


np.random.seed(0);

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
];


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


layer1 = Layer_Dense(4, 5);
layer2 = Layer_Dense(5, 2);
layer3 = Layer_Dense(2, 1);

layer1.forward(X);
layer2.forward(layer1.output);
layer3.forward(layer2.output);

layer3.show_result();
 