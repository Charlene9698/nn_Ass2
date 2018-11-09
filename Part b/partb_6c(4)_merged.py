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
HIDDEN_SIZE2 = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
batch_size = 128

no_epochs = 100  #originally 100
lr = 0.01
clipping_threshold = 2


tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

#is this a character model or word model???
def rnn_model(x):

    word_vectors = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE, name="cell1")
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits, word_list

# to use without gradient clipping
def rnn_model2(x):

    word_vectors2 = tf.contrib.layers.embed_sequence(
        x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list2 = tf.unstack(word_vectors2, axis=1)

    cell2 = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE2, name="cell2")
    _, encoding2 = tf.nn.static_rnn(cell2, word_list2, dtype=tf.float32)

    logits2 = tf.layers.dense(encoding2, MAX_LABEL, activation=None)

    return logits2, word_list2

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

    # applying gradient clippint
    optimizer = tf.train.AdamOptimizer(lr, name='adam1')
    gradients = optimizer.compute_gradients(entropy)
    # Gradient clipping
    grad_clipping = tf.constant(2.0, name="grad_clipping")
    clipped_grads_and_vars = []
    for grad, var in gradients:
        clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
        clipped_grads_and_vars.append((clipped_grad, var))

    # Gradient updates
    train_op = optimizer.apply_gradients(clipped_grads_and_vars)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    # without gradient clipping
    x2 = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y2_ = tf.placeholder(tf.int64)
    logits2, word_list2 = rnn_model2(x2)

    entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y2_, MAX_LABEL), logits=logits2))
    train_op2 = tf.train.AdamOptimizer(lr, name='adam2').minimize(entropy2)

    accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits2, axis=1), y2_), tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        loss_batch = []
        acc = []

        # without gradient clipping
        loss2 = []
        loss_batch2 = []
        acc2 = []

        # breaking down into batches
        N = len(x_train)
        idx = np.arange(N)

        for e in range(no_epochs):
            np.random.shuffle(idx)
            trainX_batch, trainY_batch = x_train[idx], y_train[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                word_list_, _, loss_  = sess.run([word_list, train_op, entropy], {x: trainX_batch[start:end], y_: trainY_batch[start:end]})
                loss_batch.append(loss_)

                #without clipping
                word_list2_, _, loss2_ = sess.run([word_list2, train_op2, entropy2],{x2: trainX_batch[start:end], y2_: trainY_batch[start:end]})
                loss_batch2.append(loss2_)

            loss.append(sum(loss_batch) / len(loss_batch))
            loss_batch[:] = []
            acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            #without clipping
            loss2.append(sum(loss_batch2) / len(loss_batch2))
            loss_batch2[:] = []
            acc2.append(accuracy2.eval(feed_dict={x2: x_test, y2_: y_test}))

            if e%10 == 0:
                print('With clipping, epoch: %d, entropy: %g'%(e, loss[e]))
                print('With clipping, epoch: %d, accuracy: %g' %(e, acc[e]))
                print('Without clipping. epoch: %d, entropy: %g' % (e, loss2[e]))
                print('Without clipping, epoch: %d, accuracy: %g' % (e, acc2[e]))

        pylab.figure(1)
        pylab.plot(range(len(loss)), loss)
        pylab.plot(range(len(loss2)), loss2)
        pylab.xlabel('epochs')
        pylab.ylabel('entropy')
        pylab.legend(['With gradient clipping', 'Without gradient clipping'])
        pylab.savefig('figures/partb_6c(4)_entropy_merged.png')

        pylab.figure(2)
        pylab.plot(range(len(acc)), acc)
        pylab.plot(range(len(acc2)), acc2)
        pylab.xlabel('epochs')
        pylab.ylabel('accuracy')
        pylab.legend(['With gradient clipping', 'Without gradient clipping'])
        pylab.savefig('figures/partb_6c(4)_accuracy_merged.png')

        pylab.show()
  
if __name__ == '__main__':
    main()
