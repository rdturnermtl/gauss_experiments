import numpy as np

N = 1000
df = 10000

X = np.random.randn(N, 2)
X = X / np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))

R = np.random.chisquare(df, size=(N, 1))
Y = X * R

R2 = np.random.power(df, size=(N, 1))
Z = X * R2

import matplotlib.pyplot as plt

plt.plot(Y[:, 0], Y[:, 1], 'b.')
plt.plot(Z[:, 0], Z[:, 1], 'r.')
