import matplotlib.pyplot as plt;
import nnfs;
from nnfs.datasets import spiral_data, vertical_data;

nnfs.init();

X, y = spiral_data(samples=100, classes = 3);

X2, y2 = vertical_data(samples=100, classes=3);

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg');
plt.show();

plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=40, cmap='brg');
plt.show();