import numpy as np;
from Layer_Dense import Layer_Dense

 
def main():
    dense = Layer_Dense(2, 4);

    print('Weights 0-1')
    print(dense.weights)
    print()

    print('Weights 1-2')
    print(dense.weights2)
    print()

    input_set_for_training = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_set_for_training = np.array([[0,1,1,0]]).T

    dense.train(input_set_for_training, output_set_for_training, 5000);

    print('Output result for testing data = ', dense.predict(np.array([1,1])));

if __name__ == "__main__":
    main()