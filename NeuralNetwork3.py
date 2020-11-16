import csv
import numpy as np

train_image = open('../data/train_image.csv')
train_label = open('../data/train_label.csv')

train_image_reader = csv.reader(train_image, delimiter=',')
train_label_reader = csv.reader(train_label, delimiter=',')

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

line_count = 0
for row in train_image_reader:
    if line_count == 0:
        print(row)
        break

for label in train_label_reader:
    if line_count == 0:
        print(label)
        break
