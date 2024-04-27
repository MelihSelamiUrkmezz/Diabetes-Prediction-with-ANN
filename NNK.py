import numpy as np;
from sklearn.model_selection import train_test_split



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


class NNK:
    
    weights: np.ndarray[np.dtype[np.float64], np.dtype[np.float64]] = None;
    weights2: np.ndarray[np.dtype[np.float64], np.dtype[np.float64]] = None;
    biases : np.ndarray[any, np.float64] = None;
    output: np.ndarray[any, np.dtype[np.float64]] = None;
    sigmoid = Sigmoid();

    n_inputs :int = 0;
    n_neurons : int = 0;

    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        
        self.n_inputs = n_inputs;
        self.n_neurons = n_neurons;

        self.weights = 2 * np.random.random((n_inputs, n_neurons)) - 1;
        self.weights2 = 2 * np.random.random((n_neurons, 1)) -1;

    def train(self, input: np.ndarray, output: np.ndarray, apochs: int):

        output = output.reshape(-1 , 1)

        for apoch in range(apochs):
            
            # print("input shape : " , input.shape); (762, 8)

            layer_1 = self.sigmoid.forward(np.dot(input, self.weights));

            # print("Layer 1 shape : " , layer_1.shape); (762, 4)

            layer_2 = self.sigmoid.forward(np.dot(layer_1, self.weights2));

            # print("Layer 2 shape  : " , layer_2.shape); (762, 1)


            layer_2_error = output - layer_2; # (762, 1) (762, 1)
            delta_2 = layer_2_error * self.sigmoid.derivative(layer_2); #(762, 1) #(762, 1)
            
            if apoch % 500 == 0:
                print('Final error after', apoch, 'iterations =', np.mean(np.abs(layer_2_error)));

            layer_1_error = np.dot(delta_2, self.weights2.T);     # (762, 1) (1, 4)
            delta_1 = layer_1_error * self.sigmoid.derivative(layer_1);  #(762, 4) (762, 4)

            # :)
            self.weights2 += np.dot(layer_1.T, delta_2);    # (4, 762)  (762, 1)
            self.weights += np.dot(input.T, delta_1); # (8, 762) (762 4)
    

    def predict(self, input: np.ndarray[any, np.dtype[np.float64]]) -> np.ndarray[any, np.dtype[np.float64]]:
        # for now we use 1 hidden layer
        layer_1 = self.sigmoid.forward(np.dot(input, self.weights));
        layer_2 = self.sigmoid.forward(np.dot(layer_1, self.weights2));

        return layer_2;




inputs = np.genfromtxt('diabetes.csv', delimiter=',', skip_header=1, usecols=[1,2,3,4,5,6,7,8])

labels = np.genfromtxt('diabetes.csv', delimiter=',', skip_header=1, usecols=-1, dtype=None, encoding=None)

# Veriyi test ve eğitim olarak bölmek
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=42)

nnk = NNK(len(inputs[0]), 4);

nnk.train(X_train, y_train, 200);


predict = nnk.predict(X_test);

print("Predict Verisi");
print(predict);

print("\nTest verisi:");
print(y_test);