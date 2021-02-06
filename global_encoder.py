import argparse
import io
import itertools
import numpy as np
import pickle

WORD_MAP_FILE = "global_word_map.pkl"

def load_corpus(filenames):
    corpus = []
    for filename in filenames:
        with io.open(filename, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                corpus.append(line.split())
    return corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", "-c", required=True, action="store", \
        help = "Corpus of tweets to load.", nargs='+')
    args = parser.parse_args()
    # Load corpus
    corpus = load_corpus(args.corpus)
    # Assemble tokenizer
    all_words = list(itertools.chain.from_iterable(corpus))
    unique_words = np.unique(all_words)
    unique_word_index = dict((c, i) for i, c in enumerate(list(unique_words)))
    with io.open(WORD_MAP_FILE, 'wb') as f:
        pickle.dump(unique_word_index, f)