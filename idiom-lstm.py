# Adapted from bit.ly/2DOGvYh

import os, re
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import *
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

vocab = np.array([chr(i) for i in range(2048)])
encoder = LabelEncoder().fit(vocab)

# For reproducibility
np.random.seed(20)

def makeDF(data_path, samples):
    '''
    Inputs:
    data_path - the path of the txt directory containing the samples
    samples - number of text samples to take for each language

    Returns:
    A pandas dataframe with the string and the language

    '''
    languages = next(os.walk('./txt'))[1]

    strings = np.array([])
    langs = np.array([])

    for language in tqdm(languages):
        file_list = glob.glob(os.path.join(data_path, "txt", language,"*.txt"))
        the_corpus = corpus(np.random.choice(file_list, samples))
        strings = np.concatenate([strings, the_corpus])
        langs = np.concatenate([langs, [language]*samples])

    return pd.DataFrame(np.array([strings, langs]).T, columns=['string','language'])

train = makeDF(os.getcwd(),100)
train['as_numbers'] = train['string'].apply(lambda x: encoder.transform(list(x)))
train['length'] = train['string'].apply(lambda x: len(x))

class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.iloc[self.cursor:self.cursor+n-1]
        self.cursor += n
        return res['as_numbers'], res['language']

class PaddedDataIterator(SimpleDataIterator):

    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.loc[self.cursor:self.cursor+n-1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['language'], res['length']

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(
    vocab_size = len(vocab),
    state_size = 64,
    batch_size = 256,
    num_classes = 21):

    reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(1.0)

    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # RNN
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size],
                                 initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen,
                                                 initial_state=init_state)

    # Add dropout, as the model otherwise quickly overfits
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    """
    Obtain the last relevant output. The best approach in the future will be to use:

        last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

    which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
    gradient for this op has not been implemented as of this writing.

    The below solution works, but throws a UserWarning re: the gradient.
    """
    idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = logits, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }

def train_graph(graph, batch_size = 256, num_epochs = 10, iterator = PaddedDataIterator):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tr = iterator(train)
        te = iterator(test)

        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                #eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                te_losses.append(accuracy / step)
                step, accuracy = 0,0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

    return tr_losses, te_losses



