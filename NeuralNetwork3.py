import csv
import numpy as np

train_image = open('../data/train_image.csv')
train_label = open('../data/train_label.csv')

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def cost_function(y, y_hat):
    m = y.shape[1]
    return -(1./m) * (np.sum(np.multiply(np.log(y_hat), y)) + np.sum(np.multiply(np.log(1-y_hat), (1-y))))

X = np.genfromtxt('../data/train_image.csv', delimiter=',')
Y = np.genfromtxt('../data/train_label.csv', delimiter=',')

Y = np.reshape(Y, (-1, 1))

X = (X/255).T

y_new = np.zeros(Y.shape)
y_new[np.where(Y == 0.0)[0]] = 1
Y = y_new

Y = Y.T

n_x = X.shape[0]
n_h = 64
learning_rate = 1

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(1, n_h)
b2 = np.zeros((1, 1))

m = 60000
cost = 0

for i in range(2000):

    Z1 = np.matmul(W1, X) + b1
    A1 = sigmoid_function(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid_function(Z2)

    cost = cost_function(Y, A2)

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_function(Z1) * (1 - sigmoid_function(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if i % 100 == 0:
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)