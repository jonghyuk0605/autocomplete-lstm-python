#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import cPickle as pickle

import models
import utils

prefix = u"안녕하세요. "
text_modeling = 'chr'
seq_len = 100
hidden_size = 128
vocab_dir = 'data/korean-english-park.train.ko.chr.vocab.pkl'

prefix = u" "+prefix

with open(vocab_dir, 'rb') as f:
    vocab = pickle.load(f)
    inv_vocab = pickle.load(f)

model = models.CHAR_RNN(hidden_size, len(vocab))

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, 'save/model')

prefix = utils.split_raw(prefix, text_modeling)
prefix = np.array(list(map(vocab.get, prefix)))

current_state = model.initial_rnn_state(1)
output = []
for x in prefix[:-1]:
    output.append(x)
    y, current_state = model.sample_output(sess, x, current_state)

x = prefix[-1]
for _ in range(seq_len - len(prefix)):
    x, current_state = model.sample_output(sess, x, current_state)
    output.append(x)

output = list(map(inv_vocab.get, output))

print utils.join_data(output, text_modeling)
