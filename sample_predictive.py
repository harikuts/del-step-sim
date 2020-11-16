#!/usr/bin/python

import tensorflow as tf
import os
import string
import numpy as np

# CREATE A GLOBAL TOKENIZER

# Get the current path
cur_dir = os.path.dirname(os.path.abspath(__file__))

# Find all book text files in data directory
data_dir = os.path.join(cur_dir, "data")
all_texts = []
for  file in os.listdir(data_dir):
    if file.endswith(".txt"):
        all_texts.append(os.path.join(data_dir, file))
print("Selected texts:\n" + '\n'.join(all_texts))

# Develop a full corpus of processed words *without* removing repitions
library = {}
for filename in all_texts:
    file = open(filename, 'r', encoding='utf8')
    # Store all lines
    lines = []
    for i in file:
        lines.append(i)
    # Compile data as full string of all text
    data = ' '.join(lines).replace('\n', '').replace('\r', ''). replace('\ufeff', '')
    # Replace all punctuation with spaces to make it more readable
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    data = data.translate(translator)
    library[os.path.basename(filename)] = data


# Go on to create the global tokenizer by compiling all the books
compiled_data = " ".join(library.values())
# Remove repeated words
z = []
for i in data.split():
    if i not in z:
        z.append(i)
filtered_data = ' '.join(z)
# Tokenizer time!
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([data])
# Save tokenizer
pass

# PROCESS ONE OF THOSE DATASETS

chosen_text = list(library.keys())[0]
sequence_data = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1
# Compile sequences
sequences = []
for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
sequences = np.array(sequences)
# Set x and y
X = []
y = []
for i in sequences:
    X.append(i[0])
    y.append(i[1])
X = np.array(X)
y = tf.keras.utils.to_categorical((np.array(y)), num_classes=vocab_size)

# CREATE MODEL

def get_LSTM(vocab_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 10, input_length=1))
    model.add(tf.keras.layers.LSTM(1000, return_sequences=True))
    model.add(tf.keras.layers.LSTM(1000))
    model.add(tf.keras.layers.Dense(1000, activation="relu"))
    model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
    return model

# Callbacks, but I don't think they're required but they might help?
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')
reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)
logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

model = get_LSTM(vocab_size)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.001))
model.fit(X, y, epochs=150, batch_size=1, callbacks=[checkpoint, reduce, tensorboard_Visualization])