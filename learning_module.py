#!/usr/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import numpy as np
import os

import pdb

CUR_DIR = os.getcwd()

X_INDEX = 1
Y_INDEX = 2
D_SIZE_INDEX = 0

NUM_EPOCHS = 2

class DataError(BaseException):
    def __init__(self, error_message=None):
        self.error_message = error_message
    def __str__(self):
        return("DataError: " + self.error_message)

class Model:
    
    def __init__(self, data=None, test_data=None):
        self.data = data
        self.model = self.create_model()
        self.sharing_model = None
        self.communal_learning_rate = 1.0

        self.test_data=test_data

    # Set data
    def setData(self, data):
        self.data = data

    # Set test data
    def setTestData(self, test_data):
        self.test_data = test_data
        
    # Takes a training step (should be called train)
    def step(self):
        if self.data is not None:
            history = self.model.fit(self.data[X_INDEX], self.data[Y_INDEX], epochs=NUM_EPOCHS)
            self.sharing_model = (self.data[D_SIZE_INDEX], self.model.get_weights())
        else:
            raise DataError("Cannot train model. No data exists.")

    def test(self):
        if self.test_data is not None:
            loss, acc = self.model.evaluate(self.test_data[X_INDEX], self.test_data[Y_INDEX], verbose=1)
            return loss, acc
        else:
            raise DataError("Cannot test model. No training data exists.")
        
    # List of tuples of [data size, weights] from other nodes
    def aggregate(self, recv_list):
        # Add self to list
        recv_list.append(self.sharing_model)
        # Aggregate all weights in list, based on the ratio of their data
        sizes = [float(x[0]) for x in recv_list]
        sizes = np.array(sizes, dtype=object)
        size_ratios = sizes / sum(sizes)
        print(size_ratios) # DEBUG
        print("Computed size ratios.")
        weights = np.array([x[1] for x in recv_list], dtype=object)
        parts = []
        print("WEIGHTS:", [w[0] for w in weights], weights.shape) # DEBUG
        for i in range(len(size_ratios)):
            products = [size_ratios[i] * w for w in weights[i]]
            parts.append(products)
        parts = np.array(parts, dtype=object)
        print("Got all parts.")
        print("ADJUSTED:", [p[0] for p in parts], parts.shape) # DEBUG
        new_weights = []
        for i in range(len(parts[0])):
            stack = [part[i] for part in parts]
            new_weights.append(sum(stack))
        new_weights = np.array(new_weights, dtype=object)
        print("NEW WEIGHTS:", new_weights[0], new_weights.shape) # DEBUG
        print("Summed all parts. New weights obtained.")

        # Perform aggregation
        cur_weights = self.model.get_weights()
        cur_weights = np.array(cur_weights, dtype=object)
        print("CUR WEIGHTS:", cur_weights[0], cur_weights.shape) # DEBUG
        learned_weights = []
        for i in range(len(cur_weights)):
            stack = (1 - self.communal_learning_rate) * cur_weights[i] + self.communal_learning_rate * new_weights[i]
            learned_weights.append(stack)
        learned_weights = np.array(learned_weights, dtype=object)
        print("LEARNED WEIGHTS:", learned_weights[0], learned_weights.shape) # DEBUG
        self.model.set_weights(learned_weights)
    
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

# DataBin class used to retrieve and return data
class DataBin:
    def __init__(self, x=None, y=None):
        # Validate that the data lengths match
        try:
            assert len(x) == len(y)
        except Exception as e:
            print("Mismatching data lengths. " + str(e))
        # Build a list of entries (x, y)
        self.data = []
        for i in range(len(x)):
            self.data.append((x[i], y[i]))
        self.data_size = len(self.data)
    # Get information about what information is available
    def getSize(self):
        return(self.data_size)
    # Retrieve data from bin
    def retrieve(self, number):
        try:
            assert number <= self.data_size
        except:
            print("Requested entries exceed maximum amount. Data left: " + str(self.getSize()))
        retrieved = [self.data.pop(0) for i in range(number)]
        x = [r[0] for r in retrieved]
        y = [r[1] for r in retrieved]
        # Change them from list to numpy array
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        # Update bin data size
        self.data_size = len(self.data)
        # Report back for diagnostics
        print("Retrieved %d entries, %d are left." % (number, self.getSize()))
        # Package (data size, x features, y labels) and return
        return (number, x, y)

# Data Incubator for handling data retrieval and storage
class DataIncubator:
    def __init__(self):
        self.data_shares = {}
        self.test_shares = {}
        self.group_shares = {}
        self.leases = {}
        self.global_data = None
        pass
    # Creates DataBin out of method get_dataset() and stores it into data_shares with name
    def createDataBin(self, name, get_dataset):
        # Extract from dataset method
        x_train, x_test, y_train, y_test = get_dataset()
        # Create databin and store it
        databin = DataBin(x_train, y_train)
        self.data_shares[name] = databin
        # Store test data too
        assert len(x_test) == len(y_test)
        self.test_shares[name] = (len(x_test), x_test, y_test)
    # Retrieves from specified databin
    def retrieve(self, name, num_entries, client_name=None):
        # Check for client name (this will be required in the future)
        if client_name is not None:
            # Call retrieve method from databin
            self.leases[client_name] = self.data_shares[name].retrieve(num_entries)
            return self.leases[client_name]
        # Legacy in case client name isn't specified
        else:
            return self.data_shares[name].retrieve(num_entries)
    # Assembles group data based on list of nodes provided matched to names on the leases
    def AssembleData(self, nodelist):
        nodelist = list(nodelist)
        # Function to combine leased datasets
        def combine(data1, data2):
            data_size = data1[D_SIZE_INDEX] + data2[D_SIZE_INDEX]
            print(data1[X_INDEX].shape, data2[X_INDEX].shape)
            x = np.concatenate((data1[X_INDEX], data2[X_INDEX]))
            print(x.shape)
            y = np.concatenate((data1[Y_INDEX], data2[Y_INDEX]))
            return (data_size, x, y)
        # Start with the first value
        if nodelist[0] in self.leases.keys():
            combined_data = self.leases[nodelist[0]]
        else:
            raise DataError("DI lease name %s not valid. Currently on lease: %s" \
                % (nodelist[0], str(list(self.leases.keys()))))
        # Continue with rest of the list
        for i in range(1, len(nodelist)):
            if nodelist[i] in self.leases.keys():
                combined_data = combine(combined_data, self.leases[nodelist[i]])
            else:
                raise DataError("DI lease name %s not valid. Currently on lease: %s" \
                    % (nodelist[i], str(list(self.leases.keys()))))
        return combined_data

    # Set global data out of all currently leased data, should be called after data is leased
    def setGlobalData(self):
        self.global_data = self.AssembleData(self.leases.keys())
        
    # DEFINE DATASET FUNCTIONS HERE (must return x_train, x_test, y_train, y_test)
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


# NOT FUNCTIONAL AT THE MOMENT
class ModelIncubator:
    def __init__(self, data_ratios):
        # Load dataset into global training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_dataset()
        # Split into specific sets
        self.data_shares = self.rsplit(self.x_train, self.y_train, nonIID=True, ratios=data_ratios)
        self.test_data = (len(self.x_test), self.x_test, self.y_test)
        print("Model incubator has been generated.")
        
    # Use this function to select one of the dataset grab functions
    def get_dataset(self):
        return self.get_mnist()

    # MNIST Dataset
    def get_mnist(self):
        # Import MNIST data
        print ("\nUnpacking MNIST data...")
        # mnist = tf.keras.datasets.mnist
        # Load data into trains
        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=CUR_DIR+'/mnist.npz')
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

class BookModelIncubator:
    def __init__(self, data_ratios):
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_dataset()
        self.data_shares = self.rsplit(self.x_train, self.y_train, nonIID=True, ratios=data_ratios)
        self.test_data = (len(self.x_test), self.x_test, self.y_test)
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

# If called as main run test scripts
if __name__ == "__main__":
    DI = DataIncubator()
    DI.createDataBin("MNIST", DI.get_mnist)
    print(DI.data_shares["MNIST"].getSize())
    # Test Model A
    data = DI.retrieve("MNIST", 5000, "Model A")
    test_data = DI.test_shares["MNIST"]
    modelA = Model(data, test_data=test_data)
    modelA.step()
    modelA.test()
    print("Model A passed!")
    # Test Model B
    data = DI.retrieve("MNIST", 3000, "Model B")
    test_data = DI.test_shares["MNIST"]
    modelB = Model(data, test_data=test_data)
    modelB.step()
    modelB.test()
    print("Model B passed!")
    # Test global data
    DI.setGlobalData()
    print(DI.global_data[0])
    assert DI.global_data[0] == 8000
    print("Global/Assembly passed!")
    
