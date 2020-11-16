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
    m = x.shape[1]
    return -(1./m) * (np.sum(np.multiply(np.log(x_hat), x)) + np.sum(np.multiply(np.log(1-x_hat), (1-x))))

for row in range(2000):
    int_row = list(map(int, train_image_list[row]))
    print(np.reshape(np.array(int_row), (28, 28)))
    int_label = list(map(int, train_label_list[row]))
    print(int_label)
    break
