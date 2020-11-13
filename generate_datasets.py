#!/usr/bin/python

import argparse
import tensorflow as tf

# Parse arguments
parser = argparse.ArgumentParser(description = "Generate distributed datasets from book text files.")
parser.add_argument('-t', '--text', action = 'store', help = "Book text file.", \
    required="True")
parser.add_argument('-n', '--clients', action = 'store', type = int, \
    help = "Number of clients to distribute data to.", \
    required = "True")
parser.add_argument('-o', '--output', action = 'store', \
    help = "Target directory for output.", \
    required = True)
parser.add_argument('--iid', action = 'store_true', default = False, \
    help = "Set flag for identically and independentally distributed data.")

args = parser.parse_args()

# PRE-PROCESSING

# Open file and construct data
# Sourced from https://towardsdatascience.com/next-word-prediction-with-nlp-and-deep-learning-48b9fe0a17bf 
file = open(args.text, 'r', encoding = 'utf8')
book_fulltext = []
for i in file:
    book_fulltext.append(i)

# Split the book
# if args.iid:
if True:
    div = len(book_fulltext) / args.clients
    book_splits = []
    ind = 0
    for i in range(args.clients):
        book_splits.append(book_fulltext[ind:ind+div])
        ind = ind + div

ind = 0
# you know, the actual process of getting these into datasets
for lines in book_splits:
    ind += 1
    # Split the dataset and process each one individually
    data = ""
    for i in lines:
        data = ' '.join(lines)
    data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')

    # Use translator to get readable data
    translator = str.maketrans(string.punctuation, ' '*(string.punctuation))
    new_data = data.translate(translator)

    z = []

    for i in data.split():
        if i not in z:
            z.append(i)

    data = ' '.join(z)

    # Tokenizer
    tokenizer = tf.keras.Tokenizer()
    tokenizer.fit_on_texts([data])

    
