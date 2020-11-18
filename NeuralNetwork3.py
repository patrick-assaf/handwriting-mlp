import time
start_time = time.time()

import numpy as np

train_image_data = np.genfromtxt('../data/train_image.csv', delimiter=',')
test_image_data = np.genfromtxt('../data/test_image.csv', delimiter=',')
train_labels = np.genfromtxt('../data/train_label.csv', delimiter=',')

digits = 10
sample_size = 60000
learning_rate = 4
beta = 0.9
batch_size = 128
number_of_batches = 469

def process_image_data(image_data):
    image_data = (image_data/255).T
    return image_data

def process_train_labels(label_data):
    label_data = np.reshape(label_data, (-1, 1))
    train_examples = label_data.shape[0]
    new_label_data = np.eye(digits)[label_data.astype('int32')]
    new_label_data = new_label_data.T.reshape(digits, train_examples)
    label_data = new_label_data[:,:sample_size]
    return label_data

train_image_data = process_image_data(train_image_data)
test_image_data = process_image_data(test_image_data )
train_labels = process_train_labels(train_labels)

# initialization
np.random.seed(138)
W1 = np.random.randn(64, 784) * np.sqrt(1/784)
b1 = np.zeros((64, 1)) * np.sqrt(1/784)
W2 = np.random.randn(digits, 64) * np.sqrt(1/64)
b2 = np.zeros((digits, 1)) * np.sqrt(1/64) 

gradients = {"dW1": 0, "db1": 0, "dW2": 0, "db2": 0}

V_dW1 = np.zeros((64, 784))
V_db1 = np.zeros((64, 1))
V_dW2 = np.zeros((digits, 64))
V_db2 = np.zeros((digits, 1))

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def cost_function(x, x_hat):
    return -(1/x.shape[1]) * (np.sum(np.multiply(x, np.log(x_hat))) + np.sum(np.multiply((1-x), np.log(1-x_hat))))

def feedforward(image_data, W1, b1, W2, b2):
    cache = {}
    cache["Z1"] = np.matmul(W1, image_data) + b1
    cache["A1"] = sigmoid_function(cache["Z1"])
    cache["Z2"] = np.matmul(W2, cache["A1"]) + b2
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)
    return cache

def backpropagation(image_data, labels, cache, W2):
    dZ2 = cache["A2"] - labels
    dW2 = (1/current_batch_size) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1/current_batch_size) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid_function(cache["Z1"]) * (1 - sigmoid_function(cache["Z1"]))
    dW1 = (1/current_batch_size) * np.matmul(dZ1, image_data.T)
    db1 = (1/current_batch_size) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# train
for epoch in range(9):

    for batch in range(number_of_batches):

        batch_start = batch * batch_size
        batch_end = min(batch_start + batch_size, sample_size-1)
        current_batch_size = batch_end - batch_start

        image_data = train_image_data[:, batch_start:batch_end]
        labels = train_labels[:, batch_start:batch_end]

        cache = feedforward(image_data, W1, b1, W2, b2)
        gradients = backpropagation(image_data, labels, cache, W2)

        V_dW1 = (beta * V_dW1 + (1 - beta) * gradients["dW1"])
        V_db1 = (beta * V_db1 + (1 - beta) * gradients["db1"])
        V_dW2 = (beta * V_dW2 + (1 - beta) * gradients["dW2"])
        V_db2 = (beta * V_db2 + (1 - beta) * gradients["db2"])

        W1 = W1 - learning_rate * V_dW1
        b1 = b1 - learning_rate * V_db1
        W2 = W2 - learning_rate * V_dW2
        b2 = b2 - learning_rate * V_db2

    cache = feedforward(train_image_data, W1, b1, W2, b2)
    train_cost = cost_function(train_labels, cache["A2"])
    cache = feedforward(test_image_data, W1, b1, W2, b2)
    print("Epoch {}: training cost = {}".format(epoch+1 ,train_cost))

print("Done.")

from sklearn.metrics import classification_report

# Test labels for precision report
Y_test = np.genfromtxt('../data/test_label.csv', delimiter=',')
Y_test = np.reshape(Y_test, (-1, 1))
test_examples = Y_test.shape[0]
Y_test_new = np.eye(digits)[Y_test.astype('int32')]
Y_test_new = Y_test_new.T.reshape(digits, test_examples)
Y_test = Y_test_new[:,:sample_size]

cache = feedforward(test_image_data, W1, b1, W2, b2)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

print(classification_report(predictions, labels))

np.savetxt("test_predictions.csv", predictions, delimiter=",", fmt='%d')

print("--- %s seconds ---" % (time.time() - start_time))