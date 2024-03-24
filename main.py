import Layer_Dense as ld
from helper.DummyDataGenerator import spiral_data 

X, y = spiral_data(100, 3);



def main():
    print(X);
    print(y);

if __name__ == "__main__":
    main()