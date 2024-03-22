import numpy as np;

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



class Loss_CategoricalCrossentropy:
    
    def __init__(self) -> None:
        pass
    
    def forward(self, y_pred: np.ndarray[any, np.dtype[np.float64]] , y_true: np.ndarray) -> np.ndarray[any]:
        samples = len(y_pred);
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7); # for escape infite value forx log()0

        if len(y_true.shape) == 1: # for just index [0, 2]
            correct_confidences = y_pred_clipped[range(samples), y_true];
        
        elif len(y_true.shape) == 2: # for One Hot Encoding [[1, 0, 0], [0, 0, 1]]
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1);

        negative_log_likelihoods = -np.log(correct_confidences);

        return negative_log_likelihoods;

class Loss(Loss_CategoricalCrossentropy):
    sample_loss : np.ndarray[any] = None;
    data_loss : float = None;

    def __init__(self) -> None:
        super().__init__()
    
    def calculate(self, output:np.ndarray[any, np.dtype[np.float64]], y:np.ndarray) -> float:
        self.sample_loss = self.forward(output, y);
        self.data_loss = np.mean(self.sample_loss);
        return self.data_loss;
    
    def to_string(self) -> None:
        print('sample_loss', self.sample_loss);
        print('data_loss', self.data_loss);  



        




 