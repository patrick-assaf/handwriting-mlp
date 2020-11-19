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
    train_examples = label_data.shape[0]
    new_label_data = np.eye(digits)[label_data.astype('int32')]
    new_label_data = new_label_data.T.reshape(digits, train_examples)
    label_data = new_label_data[:,:sample_size]
    return label_data

train_image_data = process_image_data(train_image_data)
test_image_data = process_image_data(test_image_data )
train_labels = process_train_labels(train_labels)

np.random.seed(140)
weight_one = np.random.randn(64, 784) * np.sqrt(1/784)
weight_two = np.random.randn(digits, 64) * np.sqrt(1/64)
bias_one = np.zeros((64, 1)) * np.sqrt(1/784)
bias_two = np.zeros((digits, 1)) * np.sqrt(1/64) 

gradients = { "d_weight_one": 0, "d_weight_two": 0, "d_bias_one": 0, "d_bias_two": 0 }
variances = { "d_weight_one": np.zeros((64, 784)), "d_weight_two": np.zeros((digits, 64)), "d_bias_one": np.zeros((64, 1)), "d_bias_two": np.zeros((digits, 1)) }

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def cost_function(x, x_hat):
    return -(1/x.shape[1]) * (np.sum(np.multiply(x, np.log(x_hat))) + np.sum(np.multiply((1-x), np.log(1-x_hat))))

def feed_forward_function(image_data, weight_one, weight_two, bias_one, bias_two):
    layer_one = bias_one + np.matmul(weight_one, image_data)
    activation_one = sigmoid_function(layer_one)
    layer_two = bias_two + np.matmul(weight_two, activation_one)
    activation_two = np.exp(layer_two) / np.sum(np.exp(layer_two), axis=0)
    return { "layer_one": layer_one, "activation_one": activation_one, "layer_two": layer_two, "activation_two": activation_two }

def back_propagation_function(image_data, labels, training_information, weight_two):
    layer_one, activation_one, activation_two = training_information["layer_one"], training_information["activation_one"], training_information["activation_two"]
    d_weight_one = (1/current_batch_size) * np.matmul(np.matmul(weight_two.T, activation_two-labels) * sigmoid_function(layer_one) * (1-sigmoid_function(layer_one)), image_data.T)
    d_weight_two = (1/current_batch_size) * np.matmul(activation_two-labels, activation_one.T)
    d_bias_one = (1/current_batch_size) * np.sum(np.matmul(weight_two.T, activation_two-labels) * sigmoid_function(layer_one) * (1-sigmoid_function(layer_one)), axis=1, keepdims=True)
    d_bias_two = (1/current_batch_size) * np.sum(activation_two-labels, axis=1, keepdims=True)
    return { "d_weight_one": d_weight_one, "d_weight_two": d_weight_two, "d_bias_one": d_bias_one, "d_bias_two": d_bias_two }

def shrink_variance_values(gradients):
    d_weight_one_variance = (variances["d_weight_one"] * beta + gradients["d_weight_one"] * (1-beta))
    d_weight_two_variance = (variances["d_weight_two"] * beta + gradients["d_weight_two"] * (1-beta))
    d_bias_one_variance = (variances["d_bias_one"] * beta + gradients["d_bias_one"] * (1-beta))
    d_bias_two_variance = (variances["d_bias_two"] * beta + gradients["d_bias_two"] * (1-beta))
    return { "d_weight_one": d_weight_one_variance, "d_weight_two": d_weight_two_variance, "d_bias_one": d_bias_one_variance, "d_bias_two": d_bias_two_variance }

def get_current_batch_size(batch):
    batch_start = batch * batch_size
    batch_end = min(batch_start + batch_size, sample_size-1)
    return batch_end - batch_start

for epoch in range(digits-1):
    for batch in range(number_of_batches):
        weight_one = weight_one - variances["d_weight_one"] * rate
        weight_two = weight_two - variances["d_weight_two"] * rate
        bias_one = bias_one - variances["d_bias_one"] * rate
        bias_two = bias_two - variances["d_bias_two"] * rate
        current_batch_size = get_current_batch_size(batch)
        image_data = train_image_data[:, batch * batch_size : min(batch * batch_size + batch_size, sample_size-1)]
        labels = train_labels[:, batch * batch_size : min(batch * batch_size + batch_size, sample_size-1)]
        training_information = feed_forward_function(image_data, weight_one, weight_two, bias_one, bias_two)
        gradients = back_propagation_function(image_data, labels, training_information, weight_two)
        variances = shrink_variance_values(gradients)

test_data = feed_forward_function(test_image_data, weight_one, weight_two, bias_one, bias_two)
test_predictions = np.argmax(test_data["activation_two"], axis=0)

np.savetxt("test_predictions.csv", test_predictions, delimiter=",", fmt='%d')

print("--- %s seconds ---" % (time.time() - start_time))