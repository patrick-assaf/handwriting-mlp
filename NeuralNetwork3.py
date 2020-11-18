import time
start_time = time.time()

import numpy as np

X_train = np.genfromtxt('../data/train_image.csv', delimiter=',')
X_test = np.genfromtxt('../data/test_image.csv', delimiter=',')
Y_train = np.genfromtxt('../data/train_label.csv', delimiter=',')

digits = 10
sample_size = 60000
shuffle_index = np.random.permutation(sample_size)

def process_train_image_data(image_data):
    image_data = (image_data/255).T
    image_data = image_data[:, shuffle_index]
    return image_data

def process_test_image_data(image_data):
    image_data = (image_data/255).T
    return image_data

def process_train_labels(label_data):
    label_data = np.reshape(label_data, (-1, 1))
    train_examples = label_data.shape[0]
    new_label_data = np.eye(digits)[label_data.astype('int32')]
    new_label_data = new_label_data.T.reshape(digits, train_examples)
    label_data = new_label_data[:,:sample_size]
    label_data = label_data[:, shuffle_index]
    return label_data

X_train = process_train_image_data(X_train)
X_test = process_test_image_data(X_test)
Y_train = process_train_labels(Y_train)

# hyperparameters
learning_rate = 4
beta = 0.9
batch_size = 128
batches = 469

# initialization
gradients = { 
    "dW1": 0, 
    "db1": 0, 
    "dW2": 0, 
    "db2": 0 
}

np.random.seed(138)
parameters = { 
    "W1": np.random.randn(64, 784) * np.sqrt(1/784),
    "b1": np.zeros((64, 1)) * np.sqrt(1/784),
    "W2": np.random.randn(digits, 64) * np.sqrt(1/64),
    "b2": np.zeros((digits, 1)) * np.sqrt(1/64) 
}

V_dW1 = np.zeros(parameters["W1"].shape)
V_db1 = np.zeros(parameters["b1"].shape)
V_dW2 = np.zeros(parameters["W2"].shape)
V_db2 = np.zeros(parameters["b2"].shape)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def cost_function(x, x_hat):
    return -(1/x.shape[1]) * (np.sum(np.multiply(x, np.log(x_hat))) + np.sum(np.multiply((1-x), np.log(1-x_hat))))

def feedforward(image_data, parameters):
    cache = {}
    cache["Z1"] = np.matmul(parameters["W1"], image_data) + parameters["b1"]
    cache["A1"] = sigmoid_function(cache["Z1"])
    cache["Z2"] = np.matmul(parameters["W2"], cache["A1"]) + parameters["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)
    return cache

def backpropagation(image_data, labels, parameters, cache):
    dZ2 = cache["A2"] - labels
    dW2 = (1/m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1/m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(parameters["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid_function(cache["Z1"]) * (1 - sigmoid_function(cache["Z1"]))
    dW1 = (1/m_batch) * np.matmul(dZ1, image_data.T)
    db1 = (1/m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# train
for epoch in range(9):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for batch in range(batches):

        begin = batch * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feedforward(X, parameters)
        gradients = backpropagation(X, Y, parameters, cache)

        V_dW1 = (beta * V_dW1 + (1 - beta) * gradients["dW1"])
        V_db1 = (beta * V_db1 + (1 - beta) * gradients["db1"])
        V_dW2 = (beta * V_dW2 + (1 - beta) * gradients["dW2"])
        V_db2 = (beta * V_db2 + (1 - beta) * gradients["db2"])

        parameters["W1"] = parameters["W1"] - learning_rate * V_dW1
        parameters["b1"] = parameters["b1"] - learning_rate * V_db1
        parameters["W2"] = parameters["W2"] - learning_rate * V_dW2
        parameters["b2"] = parameters["b2"] - learning_rate * V_db2

    cache = feedforward(X_train, parameters)
    train_cost = cost_function(Y_train, cache["A2"])
    cache = feedforward(X_test, parameters)
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

cache = feedforward(X_test, parameters)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

print(classification_report(predictions, labels))

np.savetxt("test_predictions.csv", predictions, delimiter=",", fmt='%d')

print("--- %s seconds ---" % (time.time() - start_time))