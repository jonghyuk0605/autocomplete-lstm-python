import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnn

class CHAR_RNN:
    # gets time series for training
    # be careful of dealing with batch/time series
    def __init__(self, hidden_size, n_vocab, n_layers = 3, w_init=tf.random_normal_initializer(), b_init=tf.constant_initializer(0), use_peepholes = True):
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        with tf.variable_scope("CHAR_RNN", reuse=False):
            self.x = tf.placeholder(tf.int32, shape = (None, None, ))
            self.y = tf.placeholder(tf.int32, shape = (None, None, ))
            self.keep_prob = tf.placeholder(tf.float32) # for dropout
            self.init_state = tf.placeholder(tf.float32, shape = (n_layers, 2, None, hidden_size))

            n_batch = tf.shape(self.x)[0]
            n_time = tf.shape(self.x)[1]
            self.onehot_x = tf.reshape(tf.one_hot(self.x, n_vocab), [-1, n_vocab])

            # embedding
            n_embed = n_vocab / 2
            with tf.variable_scope("embedding"):
                self.embd_layer = {'W': tf.get_variable("W", [n_vocab, n_embed], initializer=w_init),
                                   'b': tf.get_variable("b", [n_embed],          initializer=b_init)}
                self.embd_x = tf.matmul(self.onehot_x, self.embd_layer['W']) + self.embd_layer['b']

            state_per_layer_list = tf.unstack(self.init_state, axis = 0)
            rnn_tuple_state = tuple(
                [rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                for idx in range(n_layers)] )

            with tf.variable_scope("rnn"):
                rnn_in = tf.reshape(self.embd_x, [n_batch, n_time, n_embed])
                cell = rnn.MultiRNNCell([
                       rnn.DropoutWrapper(
                       rnn.LSTMCell(hidden_size, state_is_tuple=True, use_peepholes=use_peepholes),
                       output_keep_prob=self.keep_prob)
                       for _ in range(n_layers)],
                       state_is_tuple=True)
                self.rnn_output, self.rnn_state = tf.nn.dynamic_rnn(cell, rnn_in, initial_state=rnn_tuple_state)

            # flatten batch_size x seq_len
            self.rnn_output = tf.reshape(self.rnn_output, [-1, hidden_size])
            self.flat_y = tf.reshape(self.y, [-1])

            # compute output values
            with tf.variable_scope("output"):
                self.output_layer = {'W': tf.get_variable("W", [hidden_size, n_vocab], initializer=w_init),
                                     'b': tf.get_variable("b", [n_vocab],            initializer=b_init)}
                self.output_logprob = tf.matmul(self.rnn_output, self.output_layer['W']) + self.output_layer['b']
                self.y_ = tf.nn.softmax(self.output_logprob)

            # compute loss
            with tf.variable_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.output_logprob, labels = self.flat_y)
                self.loss = tf.reduce_mean(self.loss)

    def initial_rnn_state(self, batch_size):
        return np.zeros([self.n_layers, 2, batch_size, self.hidden_size])

    def sample_output(self, sess, x, current_state, deterministic = False):
        # single batch, time
        v = sess.run([self.y_, self.rnn_state], feed_dict={self.x: [[x]], self.init_state: current_state, self.keep_prob: 1.})
        y_prob, current_state = v[0][0], v[1]
        n_vocab = len(y_prob)
        if deterministic:
            y = np.argmax(y_prob)
        else:
            y = np.random.choice(range(n_vocab), p = y_prob)
        return y, current_state, y_prob

    def run_train_op(self, sess, train_op, batch_x, batch_y, current_state, dropout = 0.5):
        _, loss = sess.run([train_op, self.loss], feed_dict={self.x: batch_x, self.y: batch_y, self.init_state: current_state, self.keep_prob: dropout})
        return loss
