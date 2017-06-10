#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import cPickle as pickle

import models
import utils

class GhostWriter:
    def __init__(self, text_modeling, model_dir, vocab_dir, model_info):
        self.text_modeling = text_modeling
        with open(vocab_dir, 'rb') as f:
            self.vocab = pickle.load(f)
            self.inv_vocab = pickle.load(f)
        self.model = models.CHAR_RNN(model_info['hidden_size'], len(self.vocab))
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_dir)

    def random_sentence(self, prefix = u"", seq_length = 100):
        prefix = u" "+prefix
        prefix = utils.split_raw(prefix, self.text_modeling)
        prefix = np.array(list(map(self.vocab.get, prefix)))
        current_state = self.model.initial_rnn_state(1)
        output = []
        for x in prefix[:-1]:
            output.append(x)
            _, current_state, _ = self.model.sample_output(self.sess, x, current_state)
        x = prefix[-1]
        output.append(x)
        while len(output) < seq_length:
            x, current_state, _ = self.model.sample_output(self.sess, x, current_state)
            output.append(x)
        output = list(map(self.inv_vocab.get, output))
        output_str = utils.join_data(output, self.text_modeling)
        return output_str

    def sample_topk(self, prefix, end_tag = [u' ', u'.', u',', u'\n', u'\t'], k = 5, B = 100, N = 30):
        # B, N: width, depth of beam search
        prefix = u" "+prefix
        prefix = utils.split_raw(prefix, self.text_modeling)
        prefix = np.array(list(map(self.vocab.get, prefix)))
        current_state = self.model.initial_rnn_state(1)
        for x in prefix[:-1]:
            _, current_state, _ = self.model.sample_output(self.sess, x, current_state)
        end_tag = [self.vocab.get(e, -1) for e in end_tag]

        # not so efficient beam search. just initial version
        candidates = [{'p': 1., 'seq': prefix[1:], 'rnn_state': current_state, 'done': False}]
        for depth in range(len(prefix)-1, N):
            new_candidates = []
            for d in candidates:
                if d['done']:
                    new_candidates.append(d)
                    continue
                _, next_rnn_state, next_x_prob = self.model.sample_output(self.sess, d['seq'][-1], d['rnn_state'])
                for next_x in range(len(next_x_prob)):
                    next_p = d['p'] * next_x_prob[next_x]
                    next_seq = np.concatenate((d['seq'],[next_x]))
                    done = next_x in end_tag
                    new_candidates.append({'p': next_p, 'seq': next_seq, 'rnn_state': next_rnn_state, 'done': done})
            candidates = []
            new_candidates.sort(key = lambda x: x['p'], reverse = True)
            for idx in range(min(B, len(new_candidates))):
                candidates.append(new_candidates[idx])
        result = []
        for idx in range(min(k, len(candidates))):
            d = candidates[idx]
            seq = list(map(self.inv_vocab.get, d['seq']))
            result.append( (utils.join_data(seq, self.text_modeling), d['p']) )
        return result

if __name__ == '__main__':
    text_modeling = 'chr'
    model_dir = 'pretrained/model_0.ckpt'
    vocab_dir = 'pretrained/vocab.pkl'
    model_info = {'hidden_size':128}
    prefix = u""
    gw = GhostWriter(text_modeling, model_dir, vocab_dir, model_info)
    print gw.random_sentence(prefix)
    result = gw.sample_topk(u"겨")
    for s, p in result:
        print "[",
        print s,
        print "] {:.6f}".format(p)
