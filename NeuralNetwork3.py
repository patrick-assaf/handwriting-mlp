import time
start_time = time.time()

import numpy as np

train_image_data = np.genfromtxt('../data/train_image.csv', delimiter=',')
test_image_data = np.genfromtxt('../data/test_image.csv', delimiter=',')
train_labels = np.genfromtxt('../data/train_label.csv', delimiter=',')

digits = 10
sample_size = 60000
rate = 4
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

np.random.seed(140)
W1 = np.random.randn(64, 784) * np.sqrt(1/784)
W2 = np.random.randn(digits, 64) * np.sqrt(1/64)
b1 = np.zeros((64, 1)) * np.sqrt(1/784)
b2 = np.zeros((digits, 1)) * np.sqrt(1/64) 

gradients = { "dW1": 0, "db1": 0, "dW2": 0, "db2": 0 }

variances = { "dW1": np.zeros((64, 784)), "db1": np.zeros((64, 1)), "dW2": np.zeros((digits, 64)), "db2": np.zeros((digits, 1)) }

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def cost_function(x, x_hat):
    return -(1/x.shape[1]) * (np.sum(np.multiply(x, np.log(x_hat))) + np.sum(np.multiply((1-x), np.log(1-x_hat))))

def feedforward(image_data, W1, b1, W2, b2):
    Z1 = np.matmul(W1, image_data) + b1
    A1 = sigmoid_function(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
    return { "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2 }

def backpropagation(image_data, labels, cache, W2):
    Z1, A1, A2 = cache["Z1"], cache["A1"], cache["A2"]
    dW2 = (1/current_batch_size) * np.matmul(A2-labels, A1.T)
    db2 = (1/current_batch_size) * np.sum(A2-labels, axis=1, keepdims=True)
    dW1 = (1/current_batch_size) * np.matmul(np.matmul(W2.T, A2-labels) * sigmoid_function(Z1) * (1-sigmoid_function(Z1)), image_data.T)
    db1 = (1/current_batch_size) * np.sum(np.matmul(W2.T, A2-labels) * sigmoid_function(Z1) * (1-sigmoid_function(Z1)), axis=1, keepdims=True)
    return { "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2 }

def shrinkVariance(gradients):
    dW1_variance = (beta * variances["dW1"]  + (1 - beta) * gradients["dW1"])
    db1_variance = (beta * variances["db1"]  + (1 - beta) * gradients["db1"])
    dW2_variance = (beta * variances["dW2"] + (1 - beta) * gradients["dW2"])
    db2_variance = (beta * variances["db2"]  + (1 - beta) * gradients["db2"])
    return { "dW1": dW1_variance, "db1": db1_variance, "dW2": dW2_variance, "db2": db2_variance }

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
        variances = shrinkVariance(gradients)

        W1 = W1 - variances["dW1"] * rate
        b1 = b1 - variances["db1"] * rate
        W2 = W2 - variances["dW2"] * rate
        b2 = b2 - variances["db2"] * rate

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