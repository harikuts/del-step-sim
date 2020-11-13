#!/usr/bin/python

import argparse

# Parse arguments
parser = argparse.ArgumentParser(description = "Generate distributed datasets from book text files.")
parser.add_argument('-t', '--text', action = 'store', help = "Book text file.", \
    required="True")
parser.add_argument('-n', '--clients', action = 'store', \
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
lines = []
for i in file:
    lines.append(i)
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