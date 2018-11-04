
#
# Project 2 Part b
#

import math
import tensorflow as tf
import numpy as np
import pickle
from sklearn import preprocessing
import multiprocessing as mp
from functools import partial

# Proceed to the data location

import os
os.chdir('/Users/Charlene/Desktop/REP 4/Neural Networks and Deep Learning/Assignment/Assignment2 items')

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
epochs = 50
batch_size = 128
conv_window = 9  # this is the window size of convolution layer C1
conv_window2 = 5
fully_connected_layer = 300  # size of fully connected layer
pool_window = [1, 2, 2, 1]  # size of the pooling layer
pool_strides = [1, 2, 2, 1]  # strides of the pooling layer

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

with open('data_batch_1', 'rb') as fo:
    try:
        samples = pickle.load(fo)
    except UnicodeDecodeError:  # python 3.x
        fo.seek(0)
        samples = pickle.load(fo, encoding='latin1')


comeon = samples['data']
# comeon = comeon[:100]
haha = samples['labels']
# haha = haha[:100]


with open('test_batch_trim', 'rb') as fo:
    try:
        samples = pickle.load(fo)
    except UnicodeDecodeError:  # python 3.x
        fo.seek(0)
        samples = pickle.load(fo, encoding='latin1')

comeon2 = samples['data']
comeon2 = comeon[:20]
haha2 = samples['labels']
haha2 = haha[:20]


def cnn(images, nofilter_c1, nofilter_c2):
    # input data is [1, 32, 32, 3]
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # 1st Convolution Layer
    # input maps is number of channels
    # output maps should be 50
    # shape of kernel 1= [window_width, window_height, input_maps, output_maps
    W1 = tf.Variable(tf.truncated_normal([conv_window, conv_window, NUM_CHANNELS, nofilter_c1],
                                         stddev=1.0 / math.sqrt(float(NUM_CHANNELS * 9 * 9))), name='weights_1')
    # [filter_height, filter_width, in_channels, out_channels]
    # size of biases should be number of output maps
    biases_1 = tf.Variable(tf.zeros([nofilter_c1]), name='biases_1')
    # convolve the kernels and images. Number of strides is 1
    conv_1 = tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + biases_1
    # Use ReLU neurons
    conv_1_out = tf.nn.relu(conv_1)

    # 1st Pooling Layer with window of size 2 by 2, stride 2 and VALID padding, max pooling
    pool_1 = tf.nn.max_pool(conv_1_out, ksize=pool_window, strides=pool_strides, padding='VALID', name='pool_1')

    # Second convolution layer: maps 50 filters to 60 filters
    W2 = tf.Variable(tf.truncated_normal([conv_window2, conv_window2, nofilter_c1, nofilter_c2],
                                         stddev=1.0 / math.sqrt(float(nofilter_c1 * conv_window2 * conv_window2))),
                     name='weights_2')
    biases_2 = tf.Variable(tf.zeros([nofilter_c2]), name='biases_2')
    conv_2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="VALID") + biases_2
    conv_2_out = tf.nn.relu(conv_2)

    # Second pooling layer with window of size 2 by 2, stride 2 and VALID padding, max pooling
    pool_2 = tf.nn.max_pool(conv_2_out, ksize=pool_window, strides=pool_strides, padding='VALID', name='pool_2')
    dim_2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim_2])

    # Fully connected layer of size 300
    W3 = tf.Variable(tf.truncated_normal([dim_2, fully_connected_layer], stddev=1.0 / math.sqrt(float(dim_2))),
                     name='weights_3')
    biases_3 = tf.Variable(tf.zeros([fully_connected_layer]), name='biases_3')
    h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + biases_3)

    # Softmax layer
    W4 = tf.Variable(tf.truncated_normal([fully_connected_layer, NUM_CLASSES],
                                         stddev=1.0 / math.sqrt(float(fully_connected_layer))), name='weights_4')
    biases_4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.matmul(h_fc1, W4) + biases_4

    return logits, conv_1_out, pool_1, conv_2_out, pool_2


def train(nofilter_c1, nofilter_c2):
    trainX = np.array(comeon, dtype=np.float32)
    trainY_ = np.array(haha, dtype=np.int32)

    trainY = np.zeros([trainY_.shape[0], NUM_CLASSES])
    trainY[np.arange(trainY_.shape[0]), trainY_ - 1] = 1

    testX = np.array(comeon2, dtype=np.float32)
    testY_ = np.array(haha2, dtype=np.int32)

    testY = np.zeros([testY_.shape[0], NUM_CLASSES])
    testY[np.arange(testY_.shape[0]), testY_ - 1] = 1

    scaler = preprocessing.StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # define the neural network
    logits, conv_1_out, pool_1, conv_2_out, pool_2 = cnn(x, nofilter_c1, nofilter_c2)

    # calculate loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start: end]})

                batch_err.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            # Training error for one epoch: Mean batch error
            err.append(sum(batch_err) / len(batch_err))
            # accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

            print('iter', i, 'entropy', err[i])
            print ('iter', i, 'accuracy', test_acc[i])

    return test_acc[-1]


def main():

    no_threads = mp.cpu_count()

    nofilter_c1 = [30, 40, 50, 60, 70]
    nofilter_c2 = range(30, 80, 10)

    p = mp.Pool(processes=no_threads)
    result =[]
    for y in nofilter_c2:
        prod_x=partial(train,y)
        test_acc = p.map(prod_x, nofilter_c1)
        print(test_acc)
        result.append(test_acc)

    print(result)

    index=np.argmin(result)
    print('Index: {}'.format(index))
    indexc2= index/5
    optfilter_c2=nofilter_c2[indexc2]
    print('Optimal Number of Feature Maps for Second Convolution Layer: {}'.format(optfilter_c2))
    indexc1=index%5
    optfilter_c1=nofilter_c1[indexc1]
    print ('Optimal Number of Feature Maps for First Convolutional Layer: {}'.format(optfilter_c1))


if __name__ == '__main__':
    main()