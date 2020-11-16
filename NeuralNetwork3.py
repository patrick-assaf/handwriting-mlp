import csv
import numpy as np

train_image = open('../data/train_image.csv')
train_label = open('../data/train_label.csv')

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def cost_function(y, y_hat):
    m = y.shape[1]
    return -(1./m) * (np.sum(np.multiply(np.log(y_hat), y)) + np.sum(np.multiply(np.log(1-y_hat), (1-y))))

learning_rate = 1

X = np.genfromtxt('../data/train_image.csv', delimiter=',')
Y = np.genfromtxt('../data/train_label.csv', delimiter=',')

Y = np.reshape(Y, (-1, 1))

X = (X/255).T

y_new = np.zeros(Y.shape)
y_new[np.where(Y == 0.0)[0]] = 1
Y = y_new

Y = Y.T

n_x = X.shape[0]
m = X.shape[1]

W = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1, 1))

cost = 0

for i in range(2000):
    Z = np.matmul(W.T, X) + b
    A = sigmoid_function(Z)

    cost = cost_function(Y, A)

    dW = (1/m) * np.matmul(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)

    W = W - learning_rate * dW
    b = b - learning_rate * db

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)
