import os.path
import cPickle as pickle
import codecs
import collections
import numpy as np
from hangul_utils import split_syllables, join_jamos

class BatchGenerator():
    def __init__(self, text_modeling, raw_data_dir, encoding = 'utf-8'):
        vocab_dir = "{}.{}.vocab.pkl".format(raw_data_dir, text_modeling)
        data_dir = "{}.{}.data.pkl".format(raw_data_dir, text_modeling)
        if os.path.exists(data_dir):
            print "load preprocessed file {}, {}".format(vocab_dir, data_dir)
            with open(vocab_dir, 'rb') as f:
                self.vocab = pickle.load(f)
                self.inv_vocab = pickle.load(f)
            with open(data_dir, 'rb') as f:
                self.data = pickle.load(f)
        else:
            print "preprocessing file {}".format(raw_data_dir)
            with codecs.open(raw_data_dir, "r", encoding=encoding) as f:
                raw_data = f.read()
                data = split_raw(raw_data, text_modeling)
            counter = collections.Counter(data)
            count_pairs = sorted(counter.items(), key=lambda x: -x[1])
            chars, _ = zip(*count_pairs)
            self.vocab = dict(zip(chars, range(len(chars))))
            self.inv_vocab = dict(zip(range(len(chars)), chars))
            self.data = np.array(list(map(self.vocab.get, data)))
            with open(vocab_dir, 'wb') as f:
                pickle.dump(self.vocab, f)
                pickle.dump(self.inv_vocab, f)
            with open(data_dir, 'wb') as f:
                pickle.dump(self.data, f)
        self.vocab_size = len(self.vocab)
        print "Vocab size: {}".format(self.vocab_size)
        print "data length: {}".format(len(self.data))

    def get_batch(self, batch_size, seq_length):
        n_data = len(self.data)
        n_batch = n_data // (batch_size * seq_length)
        if n_batch == 0:
            print "Data is too small (or too large batch size, seq len): [{}, {}, {}]".format(n_data, batch_size, seq_length)
        for idx in range(n_batch):
            start_idx, end_idx = idx * (batch_size * seq_length), (idx+1) * (batch_size * seq_length)
            d = np.reshape(self.data[start_idx:end_idx], (batch_size, seq_length))
            yield d[:, :-1], d[:, 1:]

    def n_batch(self, batch_size, seq_length):
        n_data = len(self.data)
        return n_data // (batch_size * seq_length)

def split_raw(raw_data, text_modeling):
    if text_modeling == 'chr':
        data = []
        for c in raw_data:
            # only takes ascii or full korean syllable
            # 0x3131 ~ 0x3163: jamos, 0xac00 ~ 0xd7a3: full character
            if ord(c) < 128 or (0xac00 <= ord(c) <= 0xd7a3):
                data.extend(split_syllables(c))
    elif text_modeling == 'tok':
        pass # TODO: twitter korean tokenizer
    else:
        print 'Invalid text modeling'
    return data

def join_data(data, text_modeling):
    if text_modeling == 'chr':
        raw_data = join_jamos(data)
    elif text_modeling == 'tok':
        pass # TODO: twitter korean tokenizer
    else:
        print 'Invalid text modeling'
    return raw_data

if __name__ == "__main__":
    # test
    b = BatchGenerator('chr', 'data/korean-english-park.train.ko')
