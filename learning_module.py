#!/usr/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import numpy as np
import os

CUR_DIR = os.getcwd()

X_INDEX = 1
Y_INDEX = 2
D_SIZE_INDEX = 0

NUM_EPOCHS = 2

class Model:
    
    def __init__(self, data):
        self.data = data
        self.model = self.create_model()
        self.sharing_model = None
        self.communal_learning_rate = 1.0
        
    def step(self):
        history = self.model.fit(self.data[X_INDEX], self.data[Y_INDEX], epochs=NUM_EPOCHS)
        self.sharing_model = (self.data[D_SIZE_INDEX], self.model.get_weights())
        
    # List of tuples of [data size, weights] from other nodes
    def aggregate(self, recv_list):
        # Add self to list
        recv_list.append(self.sharing_model)
        # Aggregate all weights in list, based on the ratio of their data
        sizes = np.array([x[0] for x in recv_list])
        weights = np.array([x[1] for x in recv_list])
        weight_ratios = sizes / sum(sizes)
        new_weights = np.dot(weigh_ratios, weights)

        # Perform aggregation
        self.model = (1 - self.communal_learning_rate) * self.model + self.communal_learning_rate * new_weights
    
    # Use this function to select one of the model creation functions
    def create_model(self):
        return self.standardNN()

    # Standard Neural Network
    def standardNN(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        return model

class ModelIncubator:
    def __init__(self, data_ratios):
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_dataset()
        self.data_shares = self.rsplit(self.x_train, self.y_train, nonIID=True, ratios=data_ratios)
        print("Model incubator has been generated.")
        
    # Use this function to select one of the dataset grab functions
    def get_dataset(self):
        return self.get_mnist()

    # MNIST Dataset
    def get_mnist(self):
        # Import MNIST data
        print ("\nUnpacking MNIST data...")
#         mnist = tf.keras.datasets.mnist
        # Load data into trains
#         (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=CUR_DIR+'/mnist.npz')
        path = CUR_DIR + '/mnist.npz'
        print ("Path: ", path)
        with np.load(path) as data:
            print("Setting x_train...")
            x_train = data['x_train']
            print("\ty_train...")
            y_train = data['y_train']
            print("\tx_test...")
            x_test = data['x_test']
            print("\ty_test...")
            y_test = data['y_test']
        print ("Unpacked data!")
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, x_test, y_train, y_test

    # To split the data
    def rsplit(self, x_train, y_train, nonIID=False, ratios=None):
        # Splitting the dataset for different clients
        print ("\nSplitting data into different clients...")
        if True:
            print ("\tAssigning ranges of data...")
            accumulations = np.array([sum(ratios[0:i+1]) for i in range(len(ratios))])
            print(accumulations)
            markers = accumulations * len(x_train)
            markers = [int(marker) for marker in markers]
            print(markers)
        else:
            print ("\tUniformly assigning ranges of data")
            # markers = [1/num_clients * (n+1) for n in range(num_clients)]
        # Storing each subset of data in a client
        print ("\tStoring subsets of data into each client...")
        dataSplits = []
        for j in range(len(markers)):
            x_data = x_train[(markers[j-1] if j > 0 else 0):markers[j]]
            y_data = y_train[(markers[j-1] if j > 0 else 0):markers[j]]
            data_size = len(x_data)
            dataSplits.append((data_size, x_data, y_data))
        return dataSplits
    
# mi = ModelIncubator([0.5, 0.25, 0.25])
# m = Model(mi.data_shares[0])
# m.step()