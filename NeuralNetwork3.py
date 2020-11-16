import csv
import numpy as np

train_image = open('../data/train_image.csv')
train_label = open('../data/train_label.csv')

train_image_reader = csv.reader(train_image, delimiter=',')
train_label_reader = csv.reader(train_label, delimiter=',')

train_image_list = list(train_image_reader)
train_label_list = list(train_label_reader)

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def cost_function(y, y_hat):
    x = np.array(y)
    x_hat = np.array(y_hat)
    m = 1
    return -(1./m) * (np.sum(np.multiply(np.log(x_hat), x)) + np.sum(np.multiply(np.log(1-x_hat), (1-x))))

learning_rate = 1

X = np.array(list(map(int, train_image_list[0])))
Y = np.array(list(map(int, train_label_list[0])))

n_x = X.shape[0]
m = 1

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
