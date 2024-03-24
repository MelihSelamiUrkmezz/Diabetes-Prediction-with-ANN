import matplotlib.pyplot as plt;
from DummyDataGenerator import spiral_data 

X, y = spiral_data(100, 3);

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg');
plt.show();
