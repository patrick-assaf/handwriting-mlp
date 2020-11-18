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

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def cost_function(x, x_hat):
    return -(1/x.shape[1]) * (np.sum(np.multiply(x, np.log(x_hat))) + np.sum(np.multiply((1-x), np.log(1-x_hat))))

def feedforward(X, params):
    cache = {}
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid_function(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)
    return cache

def backpropagation(X, Y, params, cache):
    dZ2 = cache["A2"] - Y
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid_function(cache["Z1"]) * (1 - sigmoid_function(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

np.random.seed(138)

# hyperparameters
n_x = X_train.shape[0]
n_h = 64
learning_rate = 4
beta = .9
batch_size = 128
batches = -(-sample_size // batch_size)

# initialization
params = { "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
           "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
           "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
           "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h) }

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)

# train
for i in range(9):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feedforward(X, params)
        grads = backpropagation(X, Y, params, cache)

        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])

        params["W1"] = params["W1"] - learning_rate * V_dW1
        params["b1"] = params["b1"] - learning_rate * V_db1
        params["W2"] = params["W2"] - learning_rate * V_dW2
        params["b2"] = params["b2"] - learning_rate * V_db2

    cache = feedforward(X_train, params)
    train_cost = cost_function(Y_train, cache["A2"])
    cache = feedforward(X_test, params)
    print("Epoch {}: training cost = {}".format(i+1 ,train_cost))

print("Done.")

from sklearn.metrics import classification_report

# Test labels for precision report
Y_test = np.genfromtxt('../data/test_label.csv', delimiter=',')
Y_test = np.reshape(Y_test, (-1, 1))
test_examples = Y_test.shape[0]
Y_test_new = np.eye(digits)[Y_test.astype('int32')]
Y_test_new = Y_test_new.T.reshape(digits, test_examples)
Y_test = Y_test_new[:,:sample_size]

cache = feedforward(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

print(classification_report(predictions, labels))

np.savetxt("test_predictions.csv", predictions, delimiter=",", fmt='%d')

print("--- %s seconds ---" % (time.time() - start_time))