import time
start_time = time.time()

import sys
import numpy as np

train_image_data = np.genfromtxt('../data/' + sys.argv[1], delimiter=',')
train_labels = np.genfromtxt('../data/' + sys.argv[2], delimiter=',')
test_image_data = np.genfromtxt('../data/' + sys.argv[3], delimiter=',')

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
    new_label_data = np.eye(digits)[label_data.astype('int')].T.reshape(digits, label_data.shape[0])
    label_data = new_label_data[:,:sample_size]
    return label_data

train_image_data = process_image_data(train_image_data)
test_image_data = process_image_data(test_image_data )
train_labels = process_train_labels(train_labels)

gradients = { "d_weight_one": 0, "d_weight_two": 0, "d_bias_one": 0, "d_bias_two": 0 }
variances = { "d_weight_one": np.zeros((64, 784)), "d_weight_two": np.zeros((digits, 64)), "d_bias_one": np.zeros((64, 1)), "d_bias_two": np.zeros((digits, 1)) }

np.random.seed(140)
weight_one = np.sqrt(1/784) * np.random.randn(64, 784)
weight_two = np.sqrt(1/64) * np.random.randn(digits, 64)
bias_one = np.sqrt(1/784) * np.zeros((64, 1))
bias_two = np.sqrt(1/64) * np.zeros((digits, 1))

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def get_current_batch_size(batch):
    return min(sample_size-1, batch_size + batch_size * batch) - batch_size * batch

def back_propagation_function(image_data, labels, training_information, weight_two):
    layer_one, activation_one, activation_two = training_information["layer_one"], training_information["activation_one"], training_information["activation_two"]
    d_weight_one = np.matmul(np.matmul(weight_two.T, activation_two-labels) * sigmoid_function(layer_one) * (1-sigmoid_function(layer_one)), image_data.T) * (1/current_batch_size)
    d_weight_two = np.matmul(activation_two-labels, activation_one.T) * (1/current_batch_size)
    d_bias_one = np.sum(np.matmul(weight_two.T, activation_two-labels) * sigmoid_function(layer_one) * (1-sigmoid_function(layer_one)), axis=1, keepdims=True) * (1/current_batch_size)
    d_bias_two = np.sum(activation_two-labels, axis=1, keepdims=True) * (1/current_batch_size)
    return { "d_weight_one": d_weight_one, "d_weight_two": d_weight_two, "d_bias_one": d_bias_one, "d_bias_two": d_bias_two }

def feed_forward_function(image_data, weight_one, weight_two, bias_one, bias_two):
    layer_one = bias_one + np.matmul(weight_one, image_data)
    layer_two = bias_two + np.matmul(weight_two, sigmoid_function(bias_one + np.matmul(weight_one, image_data)))
    activation_one = sigmoid_function(bias_one + np.matmul(weight_one, image_data))
    activation_two = np.exp(layer_two) / np.sum(np.exp(layer_two), axis=0)
    return { "layer_one": layer_one, "activation_one": activation_one, "layer_two": layer_two, "activation_two": activation_two }

def shrink_variance_values(gradients):
    d_weight_one_variance = gradients["d_weight_one"] * (1-beta) + variances["d_weight_one"] * beta
    d_weight_two_variance = gradients["d_weight_two"] * (1-beta) + variances["d_weight_two"] * beta
    d_bias_one_variance = gradients["d_bias_one"] * (1-beta) + variances["d_bias_one"] * beta
    d_bias_two_variance = gradients["d_bias_two"] * (1-beta) + variances["d_bias_two"] * beta
    return { "d_weight_one": d_weight_one_variance, "d_weight_two": d_weight_two_variance, "d_bias_one": d_bias_one_variance, "d_bias_two": d_bias_two_variance }

for epoch in range(digits):
    for batch in range(number_of_batches):
        weight_one = weight_one - variances["d_weight_one"] * rate
        weight_two = weight_two - variances["d_weight_two"] * rate
        bias_one = bias_one - variances["d_bias_one"] * rate
        bias_two = bias_two - variances["d_bias_two"] * rate
        current_batch_size = get_current_batch_size(batch)
        image_data = train_image_data[:, batch_size * batch : min(sample_size-1, batch_size + batch_size * batch)]
        labels = train_labels[:, batch_size * batch: min(sample_size-1, batch_size + batch_size * batch)]
        training_information = feed_forward_function(image_data, weight_one, weight_two, bias_one, bias_two)
        gradients = back_propagation_function(image_data, labels, training_information, weight_two)
        variances = shrink_variance_values(gradients)

test_predictions = np.argmax(feed_forward_function(test_image_data, weight_one, weight_two, bias_one, bias_two)["activation_two"], axis=0)
np.savetxt("test_predictions.csv", test_predictions, delimiter=",", fmt='%d')

print("--- %s seconds ---" % (time.time() - start_time))