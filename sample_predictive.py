#!/usr/bin/python

import tensorflow as tf
import os

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
    data.translate(translator)
