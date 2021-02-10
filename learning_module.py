#!/usr/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
import numpy as np
import os
import random
import statistics
import pdb
import pickle
import io
from global_encoder import WORD_MAP_FILE

# Environment information
CUR_DIR = os.getcwd()
# Data object indices
X_INDEX = 1
Y_INDEX = 2
D_SIZE_INDEX = 0
# Training variables
TRAIN_TEST_SPLIT = 0.85
NUM_EPOCHS = 1
SEQ_LEN = 2
# Word map for OHE
with io.open(WORD_MAP_FILE, 'rb') as f:
    WORD_MAP = pickle.load(f)

class DataError(BaseException):
    def __init__(self, error_message=None):
        self.error_message = error_message
    def __str__(self):
        return("DataError: " + self.error_message)

class Model:
    
    def __init__(self, model_args=None):
        self.data = None
        self.model = self.LSTM(*model_args)
        self.sharing_model = None
        self.communal_learning_rate = 1.0
        self.local_test_data = None
        self.global_test_data = None

    def print_(self, message):
        # Client level print messaging
        output = "MODEL::"
        try:
            output = output + str(message)
        except:
            output = output + "Message not printable."
        print(output)

    # Set data
    def setData(self, data):
        div = int(TRAIN_TEST_SPLIT * data[D_SIZE_INDEX])
        train_x = data[X_INDEX][:div]
        test_x = data[X_INDEX][div:]
        train_y = data[Y_INDEX][:div]
        test_y = data[Y_INDEX][div:]
        self.data = (len(train_x), train_x, train_y)
        self.local_test_data = (len(test_x), test_x, test_y)

    # Set global test data.
    def setGlobalTestData(self, global_test_data):
        self.global_test_data = global_test_data
        
    # Set new learning rate
    def setLearningRate(self, val):
        self.communal_learning_rate = val
        self.print_("New learning rate is " + str(self.communal_learning_rate))

    # Takes a training step (should be called train)
    def step(self, epochs=NUM_EPOCHS):
        if self.data is not None:
            history = self.model.fit(self.data[X_INDEX], self.data[Y_INDEX], epochs=epochs)
            self.sharing_model = (self.data[D_SIZE_INDEX], self.model.get_weights())
        else:
            raise DataError("Cannot train model. No data exists.")
    # Test function, calls evaluate on passed in data; default data is loaded in test data
    def test(self, data=None):
        data = self.global_test_data if data is None else data
        if data is not None:
            loss, acc = self.model.evaluate(data[X_INDEX], data[Y_INDEX], verbose=1)
            return loss, acc
        else:
            raise DataError("Cannot test model. No testing data exists.")
        
    # List of tuples of [data size, weights] from other nodes
    def aggregate(self, recv_list):
        # Add self to list
        recv_list.append(self.sharing_model)
        # Aggregate all weights in list, based on the ratio of their data
        sizes = [float(x[0]) for x in recv_list]
        sizes = np.array(sizes, dtype=object)
        size_ratios = sizes / sum(sizes)
        # print(size_ratios) # DEBUG
        # print("Computed size ratios.")
        weights = np.array([x[1] for x in recv_list], dtype=object)
        parts = []
        # print("WEIGHTS:", [w[0] for w in weights], weights.shape) # DEBUG
        for i in range(len(size_ratios)):
            products = [size_ratios[i] * w for w in weights[i]]
            parts.append(products)
        parts = np.array(parts, dtype=object)
        # print("Got all parts.")
        # print("ADJUSTED:", [p[0] for p in parts], parts.shape) # DEBUG
        new_weights = []
        for i in range(len(parts[0])):
            stack = [part[i] for part in parts]
            new_weights.append(sum(stack))
        new_weights = np.array(new_weights, dtype=object)
        # print("NEW WEIGHTS:", new_weights[0], new_weights.shape) # DEBUG
        # print("Summed all parts. New weights obtained.")

        # Perform aggregation
        cur_weights = self.model.get_weights()
        cur_weights = np.array(cur_weights, dtype=object)
        # print("CUR WEIGHTS:", cur_weights[0], cur_weights.shape) # DEBUG
        learned_weights = []
        for i in range(len(cur_weights)):
            stack = (1 - self.communal_learning_rate) * cur_weights[i] + self.communal_learning_rate * new_weights[i]
            learned_weights.append(stack)
        learned_weights = np.array(learned_weights, dtype=object)
        # print("LEARNED WEIGHTS:", learned_weights[0], learned_weights.shape) # DEBUG
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

    def LSTM(self, vocab_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(1024, input_shape=(SEQ_LEN, vocab_size)))
        model.add(tf.keras.layers.LSTM(1024))
        model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
        return model

# DataBin class used to retrieve and return data
class DataBin:
    def __init__(self, x, y):
        # Validate that the data lengths match
        try:
            assert len(x) == len(y)
        except Exception as e:
            print("Mismatching data lengths. " + str(e))
        # Build a list of entries (x, y)
        self.data = []
        for i in range(len(x)):
            self.data.append((x[i], y[i]))
        # Shuffle this data
        random.shuffle(self.data)
        self.data_size = len(self.data)
        # Weighted shuffle the data
        # self.weightedShuffle(shuffleWeight)
        # self.verifyDistribution()
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
    # Retrieve all
    def retrieve_all(self):
        return self.retrieve(self.data_size)
    # Weighted shuffle
    def weightedShuffle(self, weight):
        # Shuffle the dataset pre-emptively
        random.shuffle(self.data)
        # Get the piles to be sorted and unsorted
        div = int(float(weight) * self.data_size)
        sorted_pile = self.data[:div]
        shuffled_pile = self.data[div:]
        # Sort the sorted pile
        def sorting_function(e):
            # Sort by y-value
            return e[1]
        sorted_pile.sort(key=sorting_function)
        # Shuffle the other pile
        random.shuffle(shuffled_pile)
        # Probabilistically reassemble the data
        data = []
        # While both piles have stuff in them
        while len(sorted_pile) > 0 and len(shuffled_pile) > 0:
            if random.random() < weight:
                data.append(sorted_pile.pop(0))
            else:
                data.append(shuffled_pile.pop(0))
        # Add the remaining of either list and done
        data = data + sorted_pile + shuffled_pile
        assert len(data) == self.data_size
        self.data = data
    def verifyDistribution(self, resolution=0.1):
        labels = [e[1] for e in self.data]
        ind = 0
        resolution = int(resolution * len(labels))
        print("Verifying distribution...")
        while ind < len(labels):
            sample = labels[ind:(ind+resolution)]
            mode = statistics.mode(sample)
            mean = statistics.mean(sample)
            print("\t%d%% - %d%% : MEAN - %d : MODE - %d" % (
                ind*100/len(labels), (ind+resolution)*100/len(labels), mean, mode
            ))
            ind = ind + resolution



# Data Incubator for handling data retrieval and storage
class DataIncubator:
    def __init__(self):
        self.data_shares = {}
        self.test_shares = {}
        self.group_shares = {}
        self.leases = {}
        self.stored_test_data = {}
        self.global_data = None
        pass
    # Creates DataBin out of method get_dataset() and stores it into data_shares with name
    def createDataBin(self, name, get_dataset, gd_args):
        # Extract from dataset method
        x, y = get_dataset(*gd_args)
        # x_train, x_test, y_train, y_test = get_dataset()
        # Create databin and store it
        databin = DataBin(x, y)
        self.data_shares[name] = databin
        # Store test data too
        # assert len(x_test) == len(y_test)
        # self.test_shares[name] = (len(x_test), x_test, y_test)
    # Retrieves from specified databin
    def retrieve(self, name, num_entries=None, client_name=None):
        # Check for client name (this will be required in the future)
        if client_name is not None:
            # Call retrieve method from databin
            if num_entries == None:
                data = self.data_shares[name].retrieve_all()
            else:
                data = self.data_shares[name].retrieve(num_entries)
            # Div up the data into train and test
            div = int(TRAIN_TEST_SPLIT * data[D_SIZE_INDEX])
            train_x = data[X_INDEX][:div]
            test_x = data[X_INDEX][div:]
            train_y = data[Y_INDEX][:div]
            test_y = data[Y_INDEX][div:]
            # Store test data
            self.stored_test_data[client_name] = (len(test_x), test_x, test_y)
            # Store all data in lease
            self.leases[client_name] = data
            return self.leases[client_name]
        # Legacy in case client name isn't specified
        else:
            return self.data_shares[name].retrieve(num_entries)
    # Assembles group data based on list of nodes provided matched to names on the leases
    def AssembleTestData(self, nodelist):
        nodelist = list(nodelist)
        # Function to combine leased datasets
        def combine(data1, data2):
            data_size = data1[D_SIZE_INDEX] + data2[D_SIZE_INDEX]
            # print(data1[X_INDEX].shape, data2[X_INDEX].shape)
            x = np.concatenate((data1[X_INDEX], data2[X_INDEX]))
            # print(x.shape)
            y = np.concatenate((data1[Y_INDEX], data2[Y_INDEX]))
            return (data_size, x, y)
        # Start with the first value
        if nodelist[0] in self.stored_test_data.keys():
            combined_data = self.stored_test_data[nodelist[0]]
        else:
            raise DataError("DI lease name %s not valid. Currently on lease: %s" \
                % (nodelist[0], str(list(self.stored_test_data.keys()))))
        # Continue with rest of the list
        for i in range(1, len(nodelist)):
            if nodelist[i] in self.stored_test_data.keys():
                combined_data = combine(combined_data, self.stored_test_data[nodelist[i]])
            else:
                raise DataError("DI lease name %s not valid. Currently on lease: %s" \
                    % (nodelist[i], str(list(self.stored_test_data.keys()))))
        return combined_data

    # Set global data out of all currently leased data, should be called after data is leased
    def setGlobalData(self):
        self.global_data = self.AssembleTestData(self.leases.keys())
        
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
        return np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))

    # Twitter Datasets
    def get_twitter_dataset(self, filename, encoder):
        # Load corpus from file
        corpus = []
        with io.open(filename, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                corpus.append(line.split())
        # Build sequences
        sequences = []
        for phrase in corpus:
            if len(phrase) >= SEQ_LEN+1:
                for i in range(SEQ_LEN, len(phrase)):
                    sequence = phrase[i-SEQ_LEN:i+1]
                    sequences.append(sequence)
        # One-hot encoding to prepare dataset
        X = np.zeros((len(sequences), SEQ_LEN, len(encoder)), dtype=bool)
        y = np.zeros((len(sequences), len(encoder)), dtype=bool)
        for i, sequence in enumerate(sequences):
            prev_words = sequence[:-1]
            next_word = sequence[-1]
            # print(prev_words, next_word)
            for j, prev_word in enumerate(prev_words):
                X[i, j, encoder[prev_word]] = 1
            y[i, encoder[next_word]] = 1
        # Return x and y
        return X, y
        
# If called as main run test scripts
if __name__ == "__main__":
    import configparser
    import matplotlib.pyplot as plt

    config = configparser.ConfigParser()
    config.read('startup-config.txt')
    encoder_file = config['TWITTER']['ENCODER']
    ds_files = config['TWITTER']['DS_FILES']
    ds_files = [n.strip() for n in ds_files.split(',')]
    num_nodes = len(ds_files)

    # Load encoder word map
    with io.open(encoder_file, 'rb') as f:
        word_map = pickle.load(f)
    # Function for assembling data
    def combine(data1, data2):
        data_size = data1[D_SIZE_INDEX] + data2[D_SIZE_INDEX]
        # print(data1[X_INDEX].shape, data2[X_INDEX].shape)
        x = np.concatenate((data1[X_INDEX], data2[X_INDEX]))
        # print(x.shape)
        y = np.concatenate((data1[Y_INDEX], data2[Y_INDEX]))
        return (data_size, x, y)
    # Create Incubator with and load data
    DI = DataIncubator()
    for f in ds_files:
        DI.createDataBin(f, DI.get_twitter_dataset, [f, word_map])
        if f is ds_files[0]:
            data = DI.retrieve(f, client_name=f)
        else:
            data = combine(data, DI.retrieve(f, client_name=f))
    # Create model and set up data
    modelA = Model(model_args=[len(word_map)])
    DI.setGlobalData()
    test_data = DI.global_data
    data = combine(data, test_data)
    # random.shuffle(data)
    # Do the learning!
    run = modelA.model.fit(data[X_INDEX], data[Y_INDEX], epochs=10, validation_split=0.15, batch_size=128, shuffle=True)
    history = run.history
    print(history.keys())
    # import pdb
    # pdb.set_trace()
    try:
        plt.plot(history['accuracy'])
    except KeyError:
        plt.plot(history['acc'])
    try:
        plt.plot(history['val_accuracy'])
    except KeyError:
        plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.show()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")
    plt.show()

        

    # DI = DataIncubator()
    # DI.createDataBin("MNIST", DI.get_mnist)
    # print(DI.data_shares["MNIST"].getSize())
    # # Test Model A
    # data = DI.retrieve("MNIST", 5000, "Model A")
    # test_data = DI.test_shares["MNIST"]
    # modelA = Model()
    # modelA.setData(data)
    # modelA.step()
    # modelA.test()
    # print("Model A passed!")
    # # Test Model B
    # data = DI.retrieve("MNIST", 3000, "Model B")
    # test_data = DI.test_shares["MNIST"]
    # modelB = Model()
    # modelB.setData(data)
    # modelB.step()
    # modelB.test()
    # print("Model B passed!")
    # # Test global data
    # DI.setGlobalData()
    # print(DI.global_data[0])
    # assert DI.global_data[0] == 8000
    # print("Global/Assembly passed!")
    
