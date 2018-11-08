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
def rnn_model(x, keep_prob):

    byte_vectors = tf.one_hot(x, no_char)
    byte_list = tf.unstack(byte_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)

    _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

    encoding = tf.nn.dropout(encoding, keep_prob)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=tf.nn.softmax)

    return logits, byte_list

def data_read_words():
  
    x_train, y_train, x_test, y_test = [], [], [], []
  
    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)

    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values

    no_char = char_processor.max_document_length
    print('Total chrarcters: %d' % no_char)

    return x_train, y_train, x_test, y_test, no_char


def main():
    global no_char

    x_train, y_train, x_test, y_test, no_char= data_read_words()

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    keep_prob = tf.placeholder(tf.float32)

    logits, word_list = rnn_model(x, keep_prob)

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
                word_list_, _, loss_  = sess.run([word_list, train_op, entropy], {x: trainX_batch[start:end], y_: trainY_batch[start:end], keep_prob:0.5})
                loss_batch.append(loss_)

            loss.append(sum(loss_batch) / len(loss_batch))
            loss_batch[:] = []
            acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob:0.5}))

            if e%10 == 0:
                print('epoch: %d, entropy: %g'%(e, loss[e]))
                print('epoch: %d, accuracy: %g' %(e, acc[e]))

        pylab.figure(1)
        pylab.plot(range(len(loss)), loss)
        pylab.xlabel('epochs')
        pylab.ylabel('entropy')
        pylab.savefig('figures/partb_5(3)_entropy.png')

        pylab.figure(2)
        pylab.plot(range(len(acc)), acc)
        pylab.xlabel('epochs')
        pylab.ylabel('accuracy')
        pylab.savefig('figures/partb_5(3)_accuracy.png')

        pylab.show()
  
if __name__ == '__main__':
    main()
