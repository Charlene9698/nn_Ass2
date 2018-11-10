import numpy as np
import pandas
import tensorflow as tf
import csv
import pylab
import time
import os

if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
batch_size = 128
embedding_size = 20

drop_out_rate = 0.5

no_epochs = 100  #originally 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

#cnn model 
def word_cnn_model(x):
    embedding = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=embedding_size)
    input_layer = tf.reshape(embedding, [-1, MAX_DOCUMENT_LENGTH, embedding_size, 1])

    #convolutional & pooling layer 1
    with tf.variable_scope('CNN_Layer1'):
        conv1 = tf.layers.conv2d(
            input_layer,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE1,
            padding='VALID',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

    #convolutional & pooling layer 2
    with tf.variable_scope('CNN_Layer2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters = N_FILTERS,
            kernel_size = FILTER_SHAPE2,
            padding='VALID',
            activation = tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size = POOLING_WINDOW,
            strides = POOLING_STRIDE,
            padding = 'SAME')

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

    #output softmax layer
    dense = tf.layers.dense(pool2, MAX_LABEL, activation=None)
    dropout = tf.layers.dropout(inputs=dense, rate=drop_out_rate)
    logits = tf.layers.dense(dropout, MAX_LABEL)
    return input_layer, logits

#cnn model without dropout
def word_cnn_model2(x):
    embedding2 = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=embedding_size)
    input_layer2 = tf.reshape(embedding2, [-1, MAX_DOCUMENT_LENGTH, embedding_size, 1])

    #convolutional & pooling layer 1
    with tf.variable_scope('CNN_Layer1_without2'):
        conv1 = tf.layers.conv2d(
            input_layer2,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE1,
            padding='VALID',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

    #convolutional & pooling layer 2
    with tf.variable_scope('CNN_Layer2_without2'):
        conv2 = tf.layers.conv2d(
            pool1,
            filters = N_FILTERS,
            kernel_size = FILTER_SHAPE2,
            padding='VALID',
            activation = tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size = POOLING_WINDOW,
            strides = POOLING_STRIDE,
            padding = 'SAME')

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

    #output softmax layer
    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)
    return input_layer2, logits


#data preprocessing 
def read_data_word():
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

    x_train, y_train, x_test, y_test, n_words = read_data_word()

    print(len(x_train))
    print(len(x_test))

    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)
    inputs, logits = word_cnn_model(x)

    #model without dropout
    x2 = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y2_ = tf.placeholder(tf.int64)
    inputs2, logits2 = word_cnn_model2(x2)

    # Optimizer
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    #optimizer without dropout
    entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y2_, MAX_LABEL), logits=logits2))
    train_op2 = tf.train.AdamOptimizer(lr).minimize(entropy2)

    #predictions
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), y_), tf.float64))

    #predictions without dropout
    accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits2, axis=1), y2_), tf.float64))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training
        loss = []
        loss_batch = []
        test_acc = []

        # training without dropout
        loss2 = []
        loss_batch2 = []
        test_acc2 = []

        #breaking down into batches
        N = len(x_train)
        idx = np.arange(N)

        training_time_1 = 0  # with dropout
        training_time_2 = 0  # without dropout
        training_time = [[], []]

        for e in range(no_epochs):
            np.random.shuffle(idx)
            trainX_batch, trainY_batch = x_train[idx], y_train[idx]

            #batch training
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                start1 = time.time()
                _, loss_ = sess.run([train_op, entropy], {x: trainX_batch[start:end], y_: trainY_batch[start:end]})
                loss_batch.append(loss_)
                training_time_1 += time.time() - start1

                #without dropout
                start2 = time.time()
                _, loss2_ = sess.run([train_op2, entropy2], {x2: trainX_batch[start:end], y2_: trainY_batch[start:end]})
                loss_batch2.append(loss2_)
                training_time_2 += time.time() - start2

            loss.append(sum(loss_batch)/len(loss_batch))
            loss_batch[:] = []
            test_acc.append(accuracy.eval(feed_dict={x: x_test, y_: y_test}))

            #without dropout
            loss2.append(sum(loss_batch2) / len(loss_batch2))
            loss_batch2[:] = []
            test_acc2.append(accuracy2.eval(feed_dict={x2: x_test, y2_: y_test}))

            if e%1 == 0:
                print('With dropout, iter: %d, entropy: %g'%(e, loss[e]))
                print('With dropout, iter: %d, accuracy: %g'%(e, test_acc[e]))
                print('Without dropout, iter: %d, entropy: %g' % (e, loss2[e]))
                print('Without dropout, iter: %d, accuracy: %g' % (e, test_acc2[e]))

        training_time[0] = training_time_1
        training_time[1] = training_time_2
        labels = ['With dropout', 'Without dropout']
        x = [0, 1]

        #plot figures here
        pylab.figure(1)
        pylab.plot(range(len(loss)), loss)
        pylab.plot(range(len(loss2)), loss2)
        pylab.xlabel('epochs')
        pylab.ylabel('entropy')
        pylab.legend(['With dropout', 'Without dropout'])
        pylab.savefig('figures/partb_5(2)_entropy_merged.png')

        pylab.figure(2)
        pylab.plot(range(len(test_acc)), test_acc)
        pylab.plot(range(len(test_acc2)), test_acc2)
        pylab.xlabel('epochs')
        pylab.ylabel('accuracy')
        pylab.legend(['With dropout', 'Without dropout'])
        pylab.savefig('figures/partb_5(2)_accuracy_merged.png')

        pylab.figure(3)
        pylab.plot(range(len(training_time)), training_time)
        pylab.ylabel('training time')
        pylab.xticks(x, labels)
        pylab.savefig('figures/partb_5(3)_trainingtime_merge2.png')

        pylab.show()
        
        sess.close()

if __name__ == '__main__':
    main()

