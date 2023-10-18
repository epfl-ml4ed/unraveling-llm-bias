import os
import json
import gzip
import struct
import argparse
import logging
from nltk import ngrams
from log import init_console_logger

LOGGER = logging.getLogger('bias scores')
LOGGER.setLevel(logging.DEBUG)


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Get the bias scores of a given text file')
    parser.add_argument('dataset_dir', help='Path to directory containing text files', type=str)
    parser.add_argument('data_source', help='reviews or suggestions', type=str)
    parser.add_argument('output_dir', help='Path to output directory', type=str)
    parser.add_argument('-n', '--num-workers', dest='num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('-w', '--window', dest='window', type=int, default=201, help='Context Window')
    return vars(parser.parse_args())


filename = 'scripts/ge-gender-pairs'
DEFAULT_FEMALE_NOUNS, DEFAULT_MALE_NOUNS = [], []
with open(filename, 'r', encoding="utf8") as f:
    gender_pairs = f.readlines()

for gp in gender_pairs:
    m, f = gp.split()
    DEFAULT_FEMALE_NOUNS.append(f)
    DEFAULT_MALE_NOUNS.append(m)


def sortbybias(d):
    d_s = sorted(d.items(), key=lambda t: t[1]['b_score'])
    return d_s


def read_vocab(vocab_path):
    """
    Read a vocabulary file. Returns a list of words
    """
    vocab = []
    with open(vocab_path, 'r', encoding="utf8") as f:
        for line in f:
            vocab.append(line.strip('\n'))

    return vocab


def read_preprocessed_file(filepath, vocab):
    """
    Reads a preprocessed text file. Returns a list of sentences, where
    each sentence is a list of tokens.
    """
    # Get binary string
    with gzip.open(filepath, 'rb') as f:
        buf = f.read()

    sentences = []
    sent = []
    for (val,) in struct.iter_unpack('I', buf):
        if val > 0:
            # Get words for the current sentence
            sent.append(vocab[val-1])
        else:
            # We've reached the end of the sentence
            sentences.append(sent)
            sent = []

    return sentences


def gender_ratios_m_f(output_data_dir, file):
    n = 0
    tot = 0
    # print("Gender Ratios...")
    with open(file, 'r', encoding="utf8") as f:
        data = json.load(f)
    bias_record = {}
    for words in data:
        if (data[words]['m']+data[words]['f'] != 0 and data[words]['f'] != 0 and data[words]['m'] != 0):
            score = data[words]['m']/(data[words]['m']+data[words]['f'])
            tot += score
            n += 1
            rec = {"b_score": score}
            data[words].update(rec)
            bias_record[words] = data[words]
    # print("bias_record", bias_record)
    # print("sortbybias", sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'biased_words_m_f')
    # print("Bias_score: ", (tot/n))
    with open(output_file, 'w', encoding="utf8") as fp:
        json.dump(bias_record, fp, sort_keys=True)


def gender_ratios(output_data_dir, file):
    # print("Gender Ratios...")
    with open(file, 'r', encoding="utf8") as f:
        data = json.load(f)
    bias_record = {}
    for words in data:
        if (data[words]['m']+data[words]['f'] != 0):
            score = data[words]['m']/(data[words]['m']+data[words]['f'])
            rec = {"b_score": score}
            data[words].update(rec)
            bias_record[words] = data[words]
    # print(bias_record)
    # print(sortbybias(bias_record))
    output_file = os.path.join(output_data_dir, 'biased_words')
    with open(output_file, 'w', encoding="utf8") as fp:
        json.dump(bias_record, fp, sort_keys=True)


def word_count(file, data):
    with open(file, 'r', encoding="utf8") as fp:
        sentences = fp.read()

    male_nouns = DEFAULT_MALE_NOUNS
    female_nouns = DEFAULT_FEMALE_NOUNS
    words = sentences.split()  # , pad_left = True, pad_right =True)

    # data = {}

    for word in words:
        if word not in data:
            data[word] = 1
        else:
            data[word] += 1

    return data


def get_cooccurrences(file, data, window):

    with open(file, 'r', encoding="utf8") as fp:
        sentences = fp.read()

    male_nouns = DEFAULT_MALE_NOUNS
    female_nouns = DEFAULT_FEMALE_NOUNS
    
    n_grams = ngrams(sentences.split(), window)
    for grams in n_grams:
        pos = 1
        m = 0
        f = 0
        for w in grams:

            pos += 1
            if w not in data:
                data[w] = {"m": 0, "f": 0}
                # print("w: ", w.encode("utf-8"))

            if pos == int((window+1)/2):
                if w in male_nouns:
                    m = 1
                    # print('word in male_nouns', w.encode("utf-8"))
                if w in female_nouns:
                    f = 1
                    # print('word in female_nouns', w.encode("utf-8"))
                if m > 0:
                    for t in grams:
                        if t not in data:
                            data[t] = {"m": 0, "f": 0}
                        data[t]['m'] += 1
                        # print("m t: ", t.encode("utf-8"))
                if f > 0:
                    for t in grams:
                        if t not in data:
                            data[t] = {"m": 0, "f": 0}
                        data[t]['f'] += 1
                        # print("f t: ", t.encode("utf-8"))
    return data


def coccurrence_counts(dataset_dir, data_source, output_dir, window=7, num_workers=1):

    output_dir = os.path.abspath('results/gender_bias_' + data_source + '/fixed/' + output_dir)
    name = dataset_dir.split('\\')[-1]
    output_data_dir = os.path.join(output_dir, name + '_bias_scores')

    if not os.path.isdir(dataset_dir):
        raise ValueError(
            'Dataset directory {} does not exist'.format(dataset_dir))

    if not os.path.isdir(output_data_dir):
        os.makedirs(output_data_dir)

    data = {}
    word_counts = {}
    worker_args = []
    LOGGER.info("Getting list of files...")
    for root, dirs, files in os.walk(dataset_dir):
        root = os.path.abspath(root)
        for fname in files:
            # only read text files
            if fname.endswith('.txt'):
                basename, ext = os.path.splitext(fname)
                if basename.lower() == 'readme':
                    continue
                txt_path = os.path.join(root, fname)
                data = get_cooccurrences(txt_path, data, window)
                word_counts = word_count(txt_path, word_counts)
    output_file = os.path.join(output_data_dir, 'all_words')
    output_file1 = os.path.join(output_data_dir, 'word_counts')

    with open(output_file, 'w', encoding="utf8") as fp:
        json.dump(data, fp)

    with open(output_file1, 'w', encoding="utf8") as fp:
        json.dump(word_counts, fp)

    gender_ratios(output_data_dir, output_file)
    gender_ratios_m_f(output_data_dir, output_file)


init_console_logger(LOGGER)
coccurrence_counts(**(parse_arguments()))
