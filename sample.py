#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import cPickle as pickle

import models
import utils

flags = tf.app.flags
flags.DEFINE_string('text_modeling', 'chr', 'chr: character-based, syl: syllable')
flags.DEFINE_string('load_dir', 'save/model', 'model to load')
flags.DEFINE_string('vocab_dir', 'data/korean-english-park.train.ko.chr.vocab.pkl', 'vocab dir')
flags.DEFINE_integer('seq_length', 100, 'length of seq to sample')
flags.DEFINE_integer('hidden_size', 128, 'hidden_size for constructing model')
args = flags.FLAGS

prefix = u"안녕하세요. "

prefix = u" "+prefix

with open(args.vocab_dir, 'rb') as f:
    vocab = pickle.load(f)
    inv_vocab = pickle.load(f)

model = models.CHAR_RNN(args.hidden_size, len(vocab))

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, args.load_dir)

prefix = utils.split_raw(prefix, args.text_modeling)
prefix = np.array(list(map(vocab.get, prefix)))

current_state = model.initial_rnn_state(1)
output = []
for x in prefix[:-1]:
    output.append(x)
    y, current_state = model.sample_output(sess, x, current_state)

x = prefix[-1]
for _ in range(args.seq_length - len(prefix)):
    x, current_state = model.sample_output(sess, x, current_state)
    output.append(x)

output = list(map(inv_vocab.get, output))

print utils.join_data(output, args.text_modeling)
