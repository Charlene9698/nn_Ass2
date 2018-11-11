#
# Project 2 Part b
#

# Import Modules
import math
import tensorflow as tf
import numpy as np
import pickle
from sklearn import preprocessing

NUM_CLASSES = 10 # number of labels
IMG_SIZE = 32
NUM_CHANNELS = 3 # number of channels in image
learning_rate = 0.001
epochs = 50
batch_size = 128
conv_window = 9  # Window size of convolution layer C1
conv_window2 = 5 # Window size of convolutional layer C2
fully_connected_layer = 300  # size of fully connected layer
pool_window = [1, 2, 2, 1]  # size of the pooling layer
pool_strides = [1, 2, 2, 1]  # strides of the pooling layer

# Initialise seed
seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# Load files
with open('data_batch_1', 'rb') as fo:
    try:
        samples = pickle.load(fo)
    except UnicodeDecodeError:  # python 3.x
        fo.seek(0)
        samples = pickle.load(fo, encoding='latin1')

trainingdata = samples['data']
# trainingdata = trainingdata[:1000]
traininglabels = samples['labels']
# traininglabels = traininglabels[:1000]

with open('test_batch_trim', 'rb') as fo:
    try:
        samples = pickle.load(fo)
    except UnicodeDecodeError:  # python 3.x
        fo.seek(0)
        samples = pickle.load(fo, encoding='latin1')

testingdata = samples['data']
# testingdata = testingdata[:200]
testinglabels = samples['labels']
# testinglabels = testinglabels[:200]

# convolutional neural networks
def cnn(images, nofilter_c1, nofilter_c2):
    # input data is [1, 32, 32, 3]
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # 1st Convolution Layer
    # input maps is number of channels, output maps should be number of feature maps
    # shape of kernel 1= [window_width, window_height, input_maps, output_maps
    W1 = tf.Variable(tf.truncated_normal([conv_window, conv_window, NUM_CHANNELS, nofilter_c1],
                                         stddev=1.0 / math.sqrt(float(NUM_CHANNELS * 9 * 9))), name='weights_1')
    # size of biases should be number of output maps
    biases_1 = tf.Variable(tf.zeros([nofilter_c1]), name='biases_1')
    # convolve the kernels and images. Number of strides is 1
    conv_1 = tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + biases_1
    # Use ReLU neurons
    conv_1_out = tf.nn.relu(conv_1)

    # 1st Pooling Layer
    # Window of size 2 by 2, stride 2 and VALID padding, max pooling
    pool_1 = tf.nn.max_pool(conv_1_out, ksize=pool_window, strides=pool_strides, padding='VALID', name='pool_1')


    # Second Convolution Layer
    # maps number of output maps in 1st Convolutional Layer to number of output maps in 2nd Convolutional Layer
    W2 = tf.Variable(tf.truncated_normal([conv_window2, conv_window2, nofilter_c1, nofilter_c2],
                                         stddev=1.0 / math.sqrt(float(nofilter_c1 * conv_window2 * conv_window2))),
                     name='weights_2')
    biases_2 = tf.Variable(tf.zeros([nofilter_c2]), name='biases_2')
    conv_2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="VALID") + biases_2
    conv_2_out = tf.nn.relu(conv_2)

    # Second Pooling Layer
    # Window of size 2 by 2, stride 2 and VALID padding, max pooling
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

# Training function
def train(nofilter_c1, nofilter_c2):

    trainX = np.array(trainingdata, dtype=np.float32)
    trainY_ = np.array(traininglabels, dtype=np.int32)

    trainY = np.zeros([trainY_.shape[0], NUM_CLASSES])
    trainY[np.arange(trainY_.shape[0]), trainY_ - 1] = 1

    testX = np.array(testingdata, dtype=np.float32)
    testY_ = np.array(testinglabels, dtype=np.int32)

    testY = np.zeros([testY_.shape[0], NUM_CLASSES])
    testY[np.arange(testY_.shape[0]), testY_ - 1] = 1

    # Scale training and test data
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
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)

    # Start a New Session
    with tf.Session() as sess:
        # Initialise the Session
        sess.run(tf.global_variables_initializer())

        test_acc = []
        err = []
        batch_err = []
        for i in range(epochs):
            # Randomly shuffle the data
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            # Use mini-batch Gradient Descent
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                # Train the Neural Network
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start: end]})

                batch_err.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            # Training error for one epoch: Mean batch error
            err.append(sum(batch_err) / len(batch_err))
            # Test accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

            # print('iter', i, 'entropy', err[i])
            # print ('iter', i, 'accuracy', test_acc[i])

    return test_acc[-1] # return the final test accuracy after 50 epochs to compare


def main():
    nofilter = range(30, 80, 10)

    result = []
    # loop through 30 to 70 feature maps for convolutional layer 1 and 2
    for y2 in range(30,80,10):
        for y1 in range(30,80,10):
            # obtain the test accuracy after every combination
            test_acc = train(y1, y2)
            result.append(test_acc)
            print('Number for y1:', y1, 'Number of y2:', y2, 'result', result)

    # obtain the index with the highest test accuracy
    index = np.argmax(result)
    print('Index: {}'.format(index))
    # obtain optimal number of feature maps for 2nd convolutional layer
    indexc2 = index // 5
    optfilter_c2 = nofilter[indexc2]
    print('Optimal Number of Feature Maps for Second Convolution Layer: {}'.format(optfilter_c2))
    # obtain optimal number of feature maps for 1st convolutional layer 
    indexc1 = index % 5
    optfilter_c1 = nofilter[indexc1]
    print ('Optimal Number of Feature Maps for First Convolutional Layer: {}'.format(optfilter_c1))


if __name__ == '__main__':
    main()