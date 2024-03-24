import numpy as np;

class Sigmoid:
    output : np.ndarray[any, np.dtype[np.float64]] = None;

    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        self.output = 1/(1 + np.exp(-1 * inputs));
        return self.output;

    def derivative(self, inputs: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        s = self.forward(inputs);
        self.output =  s * (1 - s);
        return self.output;

    def show_result(self) -> None:
        print(self.output);

class Relu:
    output : np.ndarray[any, np.dtype[np.float64]] = None;

    def __init__(self) -> None:
        pass

    def forward(self, inputs: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        self.output = np.maximum(0, inputs);
        return self.output;

    def derivative(self, inputs: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        self.output = np.where(inputs <= 0, 0, 1);
        return self.output;

    def show_result(self) -> None:
        print(self.output);

# Use for multiple classes problems
class Softmax: # exponential function with normalization
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