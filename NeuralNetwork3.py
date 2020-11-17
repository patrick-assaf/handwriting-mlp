import time
start_time = time.time()

import numpy as np

train_image = open('../data/train_image.csv')
train_label = open('../data/train_label.csv')

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def cost_function(y, y_hat):
    m = y.shape[1]
    return -(1./m) * (np.sum(np.multiply(np.log(y_hat), y)) + np.sum(np.multiply(np.log(1-y_hat), (1-y))))

def compute_multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum
    return L

X_train = np.genfromtxt('../data/train_image.csv', delimiter=',')
Y_train = np.genfromtxt('../data/train_label.csv', delimiter=',')

X_test = np.genfromtxt('../data/test_image.csv', delimiter=',')
Y_test = np.genfromtxt('../data/test_label.csv', delimiter=',')

Y_train = np.reshape(Y_train, (-1, 1))
Y_test = np.reshape(Y_test, (-1, 1))

X_train = (X_train/255).T
X_test = (X_test/255).T

digits = 10

train_examples = Y_train.shape[0]
test_examples = Y_test.shape[0]

Y_train_new = np.eye(digits)[Y_train.astype('int32')]
Y_train_new = Y_train_new.T.reshape(digits, train_examples)

Y_test_new = np.eye(digits)[Y_test.astype('int32')]
Y_test_new = Y_test_new.T.reshape(digits, test_examples)

m = 60000

Y_train = Y_train_new[:,:m]
Y_test = Y_test_new[:,:m]

shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

learning_rate = 1

n_x = X_train.shape[0]
n_h = 64

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(digits, n_h)
b2 = np.zeros((digits, 1))

X = X_train
Y = Y_train

cost = 0

for i in range(2000):

    Z1 = np.matmul(W1,X) + b1
    A1 = sigmoid_function(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

    cost = compute_multiclass_loss(Y, A2)

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

    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)

from sklearn.metrics import classification_report, confusion_matrix

Z1 = np.matmul(W1, X_test) + b1
A1 = sigmoid_function(Z1)
Z2 = np.matmul(W2, A1) + b2
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

predictions = np.argmax(A2, axis=0)
labels = np.argmax(Y_test, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))

np.savetxt("test_predictions.csv", predictions, delimiter=",", fmt='%d')

print("--- %s seconds ---" % (time.time() - start_time))