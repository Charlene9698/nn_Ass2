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

def rnn_model_basicRnn(x):

    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=tf.nn.softmax)

    return logits, word_list

#model with lstm cells
def rnn_model_lstm(x):

    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    outputs, states = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(states[-1], MAX_LABEL, activation=tf.nn.softmax)
    return logits, word_list

#model with gru cells
def rnn_model_gru(x):

    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=tf.nn.softmax)

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

    # basic rnn
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    logits, word_list = rnn_model_basicRnn(x)

    #lstm model
    x2 = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y2_ = tf.placeholder(tf.int64)
    logits2, word_list2 = rnn_model_lstm(x2)

    # gru model
    x3 = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y3_ = tf.placeholder(tf.int64)
    logits3, word_list3 = rnn_model_gru(x3)

    #basic rnn
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    #lstm model
    entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y2_, MAX_LABEL), logits=logits2))
    train_op2 = tf.train.AdamOptimizer(lr).minimize(entropy2)
    accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits2, axis=1), y2_), tf.float64))

    # gru model
    entropy3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y3_, MAX_LABEL), logits=logits3))
    train_op3 = tf.train.AdamOptimizer(lr).minimize(entropy3)
    accuracy3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits3, axis=1), y3_), tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # basic rnn
        loss = []
        loss_batch = []
        acc = []

        # lstm
        loss2 = []
        loss_batch2 = []
        acc2 = []

        # gru
        loss3 = []
        loss_batch3 = []
        acc3 = []

        # breaking down into batches
        N = len(x_train)
        idx = np.arange(N)

        for e in range(no_epochs):
            np.random.shuffle(idx)
            trainX_batch, trainY_batch = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                #basic rnn
                word_list_, _, loss_  = sess.run([word_list, train_op, entropy], {x: trainX_batch[start:end], y_: trainY_batch[start:end]})
                loss_batch.append(loss_)

                #lstm
                word_list2_, _, loss2_ = sess.run([word_list2, train_op2, entropy2],{x2: trainX_batch[start:end], y2_: trainY_batch[start:end]})
                loss_batch2.append(loss2_)

                #gru
                word_list3_, _, loss3_ = sess.run([word_list3, train_op3, entropy3],{x3: trainX_batch[start:end], y3_: trainY_batch[start:end]})
                loss_batch3.append(loss3_)

            #basic rnn
            loss.append(sum(loss_batch) / len(loss_batch))
            loss_batch[:] = []
            acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            #lstm
            loss2.append(sum(loss_batch2) / len(loss_batch2))
            loss_batch2[:] = []
            acc2.append(accuracy2.eval(feed_dict={x2: x_test, y2_: y_test}))

            #gru
            loss3.append(sum(loss_batch3) / len(loss_batch3))
            loss_batch3[:] = []
            acc3.append(accuracy3.eval(feed_dict={x3: x_test, y3_: y_test}))


            if e%10 == 0:
                print('Basic RNN, epoch: %d, entropy: %g'%(e, loss[e]))
                print('Basic RNN, epoch: %d, accuracy: %g' %(e, acc[e]))
                print('LSTM, epoch: %d, entropy: %g' % (e, loss2[e]))
                print('LSTM, epoch: %d, accuracy: %g' % (e, acc2[e]))
                print('GRU, epoch: %d, entropy: %g' % (e, loss3[e]))
                print('GRU, epoch: %d, accuracy: %g' % (e, acc3[e]))

        pylab.figure(1)
        pylab.plot(range(len(loss)), loss)
        pylab.plot(range(len(loss2)), loss2)
        pylab.plot(range(len(loss3)), loss3)
        pylab.xlabel('epochs')
        pylab.ylabel('entropy')
        pylab.legend(['Basic RNN', 'LSTM', 'GRU'])
        pylab.savefig('figures/partb_6a(4)_entropy_merged.png')

        pylab.figure(2)
        pylab.plot(range(len(acc)), acc)
        pylab.plot(range(len(acc2)), acc2)
        pylab.plot(range(len(acc3)), acc3)
        pylab.xlabel('epochs')
        pylab.ylabel('accuracy')
        pylab.legend(['Basic RNN', 'LSTM', 'GRU'])
        pylab.savefig('figures/partb_6a(4)_accuracy_merged.png')

        pylab.show()
  
if __name__ == '__main__':
    main()
