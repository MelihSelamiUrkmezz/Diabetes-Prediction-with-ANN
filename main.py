import Layer_Dense as ld
import nnfs;
from nnfs.datasets import spiral_data; # or you can use ./DummyDataGenerator for this. just uncomment underline :)
# from DummyDataGenerator import spiral_data as mySprialData

nnfs.init();

X, y = spiral_data(100, 3);

def main():
    activation_relu = ld.Activation_Relu();
    activation_softmax = ld.Activation_Softmax();

    layer1 = ld.Layer_Dense(2, 5);
    layer1.forward(X);
    activation_relu.forward(layer1.output);

    layer2 = ld.Layer_Dense(5, 3);
    layer2.forward(activation_relu.output);
    activation_softmax.forward(layer2.output);

    activation_softmax.show_result();

    loss_calculator = ld.Loss();
    loss = loss_calculator.calculate(activation_softmax.output, y);

    print('Loss', loss);

if __name__ == "__main__":
    main()