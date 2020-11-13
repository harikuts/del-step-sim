#!/usr/bin/python

import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Generate distributed datasets from book text files.")
parser.add_argument('-t', '--text', action='store', help="Book text file.", \
    required="True")
parser.add_argument('-n', '--clients', action='store', \
    help="Number of clients to distribute data to.", \
    required="True")
parser.add_argument('-o', '--output', action='store', \
    help="Target directory for output.", \
    required=True)
parser.add_argument('--iid', action='store_true', default=False, \
    help="Set flag for identically and independentally distributed data.")

args = parser.parse_args()