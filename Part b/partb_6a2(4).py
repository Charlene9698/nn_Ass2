import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab

import os


if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')


MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
batch_size = 128

no_epochs = 100  #originally 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

#is this a character model or word model???
def rnn_model(x):

    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

    outputs, states = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)
    print('here')

    logits = tf.layers.dense(states[-1], MAX_LABEL, activation=tf.nn.softmax)
    return logits, word_list

def data_read_words():
  
    x_train, y_train, x_test, y_test = [], [], [], []
  
    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))
  
    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)

    return x_train, y_train, x_test, y_test, no_words

def main():
    global n_words

    x_train, y_train, x_test, y_test, n_words = data_read_words()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    logits, word_list = rnn_model(x)

    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        loss_batch = []
        acc = []

        # breaking down into batches
        N = len(x_train)
        idx = np.arange(N)

        for e in range(no_epochs):
            np.random.shuffle(idx)
            trainX_batch, trainY_batch = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                word_list_, _, loss_  = sess.run([word_list, train_op, entropy], {x: trainX_batch[start:end], y_: trainY_batch[start:end]})
                loss_batch.append(loss_)

            loss.append(sum(loss_batch) / len(loss_batch))
            loss_batch[:] = []
            acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            if e%10 == 0:
                print('epoch: %d, entropy: %g'%(e, loss[e]))
                print('epoch: %d, accuracy: %g' %(e, acc[e]))

        pylab.figure(1)
        pylab.plot(range(len(loss)), loss)
        pylab.xlabel('epochs')
        pylab.ylabel('entropy')
        pylab.savefig('figures/partb_6a2(4)_entropy.png')

        pylab.figure(2)
        pylab.plot(range(len(acc)), acc)
        pylab.xlabel('epochs')
        pylab.ylabel('accuracy')
        pylab.savefig('figures/partb_6a2(4)_accuracy.png')

        pylab.show()
  
if __name__ == '__main__':
    main()
