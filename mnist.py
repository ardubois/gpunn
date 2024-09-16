import sys, numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:1000].reshape(1000,28*28)/255, y_train[0:1000])
one_hot_labels = np.zeros((len(labels),10))
for i,l in enumerate(labels):
	one_hot_labels[i][l] = 1


np.savetxt("mnist.csv", images, delimiter=",")

np.savetxt("labels.csv", one_hot_labels, delimiter=",")
