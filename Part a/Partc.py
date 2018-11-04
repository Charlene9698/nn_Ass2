#
# Project 2 Part a
# Need to change the number of filters after running through Part B

import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
from sklearn import preprocessing

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 50
batch_size = 128
conv_window = 9  # this is the window size of convolution layer C1
conv_window2 = 5
fully_conn_layer = 300  # size of fully connected layer
pool_window = [1, 2, 2, 1]  # size of the pooling layer
pool_strides = [1, 2, 2, 1]  # strides of the pooling layer

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1

    return data, labels_


def cnn(images, window_size, window_size2, pool_window, pool_strides, fully_connected_layer):
    # input data is [1, 32, 32, 3]
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # 1st Convolution Layer
    # input maps is number of channels
    # output maps should be 50
    # shape of kernel 1= [window_width, window_height, input_maps, output_maps
    W1 = tf.Variable(tf.truncated_normal([window_size, window_size, NUM_CHANNELS, 50],
                                         stddev=1.0 / math.sqrt(float(NUM_CHANNELS * 9 * 9))), name='weights_1')
    # [filter_height, filter_width, in_channels, out_channels]
    # size of biases should be number of output maps
    biases_1 = tf.Variable(tf.zeros([50]), name='biases_1')
    # convolve the kernels and images. Number of strides is 1
    conv_1 = tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + biases_1
    # Use ReLU neurons
    conv_1_out = tf.nn.relu(conv_1)

    # 1st Pooling Layer with window of size 2 by 2, stride 2 and VALID padding, max pooling
    pool_1 = tf.nn.max_pool(conv_1_out, ksize=pool_window, strides=pool_strides, padding='VALID', name='pool_1')

    # Second convolution layer: maps 50 filters to 60 filters
    W2 = tf.Variable(tf.truncated_normal([window_size2, window_size2, 50, 60],
                                         stddev=1.0 / math.sqrt(float(50 * window_size2 * window_size2))),
                     name='weights_2')
    biases_2 = tf.Variable(tf.zeros([60]), name='biases_2')
    conv_2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="VALID") + biases_2
    conv_2_out = tf.nn.relu(conv_2)

    # Second pooling layer with window of size 2 by 2, stride 2 and VALID padding, max pooling
    pool_2 = tf.nn.max_pool(conv_2_out, ksize=pool_window, strides=pool_strides, padding='VALID', name='pool_2')
    dim_2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim_2])

    # Fully connected layer of size 300
    W3 = tf.Variable(tf.truncated_normal([dim_2, fully_connected_layer], stddev=1.0 /
                                                                                math.sqrt(float(dim_2))),
                     name='weights_3')
    biases_3 = tf.Variable(tf.zeros([fully_connected_layer]), name='biases_3')
    h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + biases_3)

    # Softmax layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W4 = tf.Variable(tf.truncated_normal([fully_connected_layer, NUM_CLASSES],
                                                stddev=1.0 / math.sqrt(float(fully_connected_layer))),name='weights_4')
    biases_4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.matmul(h_fc1_drop, W4) + biases_4

    return logits, conv_1_out, pool_1, conv_2_out, pool_2, keep_prob


def main():
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    scaler = preprocessing.StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    logits, conv_1_out, pool_1, conv_2_out, pool_2, keep_prob = cnn(x, conv_window, conv_window2, pool_window, pool_strides, fully_conn_layer)

    # calculate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    train_step1 = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    train_step2 = tf.train.RMSPropOptimizer(1e-4).minimize(cross_entropy)
    train_step3 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    # converts tensor to new type
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_acc = []
        err = []
        batch_err = []
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start: end], keep_prob: 1.0})

                batch_err.append(loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0}))

            # Training error for one epoch: Mean batch error
            err.append(sum(batch_err) / len(batch_err))
            # accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))

            print('iter', i, 'entropy', err[i])
            print ('iter', i, 'accuracy', test_acc[i])

        plt.figure()
        plt.plot(np.arange(epochs), test_acc, label='Gradient Descent')
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')
        plt.legend(loc='lower right')

        plt.plot(np.arange(epochs), err, label = "Gradient Descent")
        plt.xlabel('epochs')
        plt.ylabel('err')

        # With Momentum
        print('momentum...')
        sess.run(tf.global_variables_initializer())

        test_acc = []
        err=[]
        batch_err=[]
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0,N,batch_size),range(batch_size,N, batch_size)):
                train_step1.run(feed_dict={x:trainX[start:end], y_:trainY[start: end], keep_prob:1.0})
                batch_err.append(loss.eval(feed_dict={x:trainX, y_:trainY, keep_prob: 1.0}))

            #Training error for one epoch: Mean batch error
            err.append(sum(batch_err)/len(batch_err))
            #accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x:testX, y_: testY, keep_prob: 1.0}))

            print('iter', i, 'entropy', err[i])
            print ('iter',i, 'accuracy',test_acc[i])

        plt.plot(np.arange(epochs), test_acc, label = "momentum")
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')

        plt.plot(np.arange(epochs), err, label = "momentum")
        plt.xlabel('epochs')
        plt.ylabel('err')

        # Using RMSProp algorithm
        print('RMS Prop...')
        sess.run(tf.global_variables_initializer())
        test_acc=[]
        err=[]
        batch_err=[]
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0,N,batch_size),range(batch_size,N, batch_size)):
                train_step2.run(feed_dict={x:trainX[start:end], y_:trainY[start: end], keep_prob:1.0})
                batch_err.append(loss.eval(feed_dict={x:trainX, y_:trainY, keep_prob:1.0}))

            #Training error for one epoch: Mean batch error
            err.append(sum(batch_err)/len(batch_err))
            #accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x:testX, y_: testY, keep_prob: 1.0}))

            print('iter', i, 'entropy', err[i])
            print ('iter',i, 'accuracy',test_acc[i])

        plt.plot(np.arange(epochs), test_acc, label = "RMS Prop")
        plt.xlabel('epochs')
        plt.ylabel('test acc')

        plt.plot(np.arange(epochs), err, label="RMS Prop")
        plt.xlabel('epochs')
        plt.ylabel('err')

        # Using Adam Optimizer
        print('Adam...')
        sess.run(tf.global_variables_initializer())
        test_acc=[]
        err=[]
        batch_err=[]
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0,N,batch_size),range(batch_size,N, batch_size)):
                train_step3.run(feed_dict={x:trainX[start:end], y_:trainY[start: end], keep_prob:1.0})
                batch_err.append(loss.eval(feed_dict={x:trainX, y_:trainY, keep_prob: 1.0}))

            #Training error for one epoch: Mean batch error
            err.append(sum(batch_err)/len(batch_err))
            #accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x:testX, y_: testY, keep_prob: 1.0}))

            print('iter', i, 'entropy', err[i])
            print ('iter',i, 'accuracy',test_acc[i])

        # Normal Gradient Descent with Dropout of 0.9
        print('gd...')
        sess.run(tf.global_variables_initializer())

        test_acc=[]
        err=[]
        batch_err=[]
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0,N,batch_size),range(batch_size,N, batch_size)):
                train_step.run(feed_dict={x:trainX[start:end], y_:trainY[start: end], keep_prob:0.9})
                batch_err.append(loss.eval(feed_dict={x:trainX, y_:trainY, keep_prob: 0.9}))

            #Training error for one epoch: Mean batch error
            err.append(sum(batch_err)/len(batch_err))
            #accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x:testX, y_: testY, keep_prob: 0.9}))

            print('iter', i, 'entropy', err[i])
            print ('iter',i, 'accuracy',test_acc[i])

        plt.figure()
        plt.plot(np.arange(epochs), test_acc, label='Gradient Descent')
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')
        plt.legend(loc='lower right')

        plt.plot(np.arange(epochs), err, label= "Gradient Descent")
        plt.xlabel('epochs')
        plt.ylabel('error')

    plt.show()


if __name__ == '__main__':
    main()