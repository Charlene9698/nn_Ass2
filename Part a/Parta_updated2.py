#
# Project 2 Part a
#

# Import Modules
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
from sklearn import preprocessing

NUM_CLASSES = 10 # number of labels
IMG_SIZE = 32
NUM_CHANNELS = 3 # number of channels of image
learning_rate = 0.001
epochs = 50
batch_size = 128 # batch size for mini-batch gradient descent
conv_window = 9  # window size of convolution layer C1
conv_window2 = 5 # window size for convolution layer C2
fully_conn_layer = 300  # size of fully connected layer
pool_window = [1, 2, 2, 1]  # size of the pooling layer
pool_strides = [1, 2, 2, 1]  # strides of the pooling layer

# initialise seed
seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

# load files
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

# convolutional neural network
def cnn(images, window_size, window_size2, pool_window, pool_strides, fully_connected_layer):
    # input data is [1, 32, 32, 3]
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # 1st Convolution Layer
    # input maps is number of channels, output maps should be 50
    # shape of kernel 1= [window_width, window_height, input_maps, output_maps]
    W1 = tf.Variable(tf.truncated_normal([window_size, window_size, NUM_CHANNELS, 50],
                                         stddev=1.0 / math.sqrt(float(NUM_CHANNELS * 9 * 9))), name='weights_1')
    # size of biases should be number of output maps
    biases_1 = tf.Variable(tf.zeros([50]), name='biases_1')
    # convolve the kernels and images. Number of strides is 1
    conv_1 = tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='VALID') + biases_1
    # Use ReLU neurons
    conv_1_out = tf.nn.relu(conv_1)

    # 1st Pooling Layer
    # Window of size 2 by 2, stride 2 and VALID padding, max pooling
    pool_1 = tf.nn.max_pool(conv_1_out, ksize=pool_window, strides=pool_strides, padding='VALID', name='pool_1')

    # 2nd Convolutional Layer
    # maps 50 filters to 60 filters
    W2 = tf.Variable(tf.truncated_normal([window_size2, window_size2, 50, 60],
                                         stddev=1.0 / math.sqrt(float(50 * window_size2 * window_size2))),
                     name='weights_2')
    biases_2 = tf.Variable(tf.zeros([60]), name='biases_2')
    conv_2 = tf.nn.conv2d(pool_1, W2, strides=[1, 1, 1, 1], padding="VALID") + biases_2
    conv_2_out = tf.nn.relu(conv_2)

    # 2nd Pooling Layer
    # Window of size 2 by 2, stride 2 and VALID padding, max pooling
    pool_2 = tf.nn.max_pool(conv_2_out, ksize=pool_window, strides=pool_strides, padding='VALID', name='pool_2')
    dim_2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim_2])

    # Fully Connected Layer of size 300
    W3 = tf.Variable(tf.truncated_normal([dim_2, fully_connected_layer], stddev=1.0 /
                                                                                math.sqrt(float(dim_2))),
                     name='weights_3')
    biases_3 = tf.Variable(tf.zeros([fully_connected_layer]), name='biases_3')
    h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W3) + biases_3)

    # Softmax layer
    W4 = tf.Variable(tf.truncated_normal([fully_connected_layer, NUM_CLASSES],
                                         stddev=1.0 / math.sqrt(float(fully_connected_layer))), name='weights_4')
    biases_4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    logits = tf.matmul(h_fc1, W4) + biases_4

    return logits, conv_1_out, pool_1, conv_2_out, pool_2


def main():
    # Training Data and Labels
    trainX, trainY = load_data('data_batch_1')
    print(trainX.shape, trainY.shape)

    # Test Data and labels
    testX, testY = load_data('test_batch_trim')
    print(testX.shape, testY.shape)

    # Scale Training and Test Data
    scaler = preprocessing.StandardScaler()
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    logits, conv_1_out, pool_1, conv_2_out, pool_2 = cnn(x, conv_window, conv_window2, pool_window, pool_strides,
                                                         fully_conn_layer)

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
        # Initialise Session
        sess.run(tf.global_variables_initializer())

        test_acc = []
        err = []
        batch_err = []

        for i in range(epochs):
            # Randomly shuffle the data
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            # Use Mini-Batch Gradient Descent
            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                # Train the Neural Network
                train_step.run(feed_dict={x: trainX[start:end], y_: trainY[start: end]})

                batch_err.append(loss.eval(feed_dict={x: trainX, y_: trainY}))

            # Training error for one epoch: Mean batch error
            err.append(sum(batch_err) / len(batch_err))
            # Test accuracy for one epoch
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

            print('iter', i, 'entropy', err[i])
            print ('iter', i, 'accuracy', test_acc[i])

        # Plot random training image
        ind = np.random.randint(low=0, high=10000)
        X = trainX[ind, :]
        plt.figure()
        plt.gray()
        X_show = X.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
        X_show = X_show.transpose(1, 2, 0)
        plt.axis('off')
        plt.imshow(X_show)

        # Plot test accuracy against epochs
        plt.figure()
        plt.plot(np.arange(epochs), test_acc)
        plt.xlabel('epochs')
        plt.ylabel('test accuracy')
        plt.legend(loc='lower right')

        # Plot training cost against epochs
        plt.figure()
        plt.plot(np.arange(epochs), err)
        plt.xlabel('epochs')
        plt.ylabel('Error')
        plt.legend(loc='lower right')

        # Pick first test pattern and plot it
        ind1 = np.random.randint(low=0, high=2000)
        X1 = testX[ind1, :]
        plt.figure()
        plt.gray()
        X1_show = X1.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
        X1_show = X1_show.transpose(1, 2, 0)
        plt.axis('off')
        plt.imshow(X1_show)

        # Pick second test pattern and plot it
        ind2 = np.random.randint(low=0, high=2000)
        X2 = testX[ind2, :]
        plt.figure()
        plt.gray()
        X2_show = X2.reshape(NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
        X2_show = X2_show.transpose(1, 2, 0)
        plt.axis('off')
        plt.imshow(X2_show)

        # Obtain the feature maps of convolutional and pooling layers for 1st test pattern
        conv_1_out_, pool_1_, conv_2_out_, pool_2_ = sess.run([conv_1_out, pool_1, conv_2_out, pool_2],
                                                              {x: X1.reshape(1, 32 * 32 * 3)})

        # Plot feature map of 1st Convolutional Layer
        plt.figure()
        plt.gray()
        conv_1_out_ = np.array(conv_1_out_)
        for i in range(50):
            plt.subplot(5, 10, i + 1);
            plt.axis('off');
            plt.imshow(conv_1_out_[0, :, :, i])

        # Plot feature map of 1st Pooling Layer
        plt.figure()
        plt.gray()
        pool_1_ = np.array(pool_1_)
        for i in range(50):
            plt.subplot(5, 10, i + 1);
            plt.axis('off');
            plt.imshow(pool_1_[0, :, :, i])

        # Plot feature map of 2nd Convolutional Layer
        plt.figure()
        plt.gray()
        conv_2_out_ = np.array(conv_2_out_)
        for i in range(60):
            plt.subplot(6, 10, i + 1);
            plt.axis('off');
            plt.imshow(conv_2_out_[0, :, :, i])

        # Plot feature map of 2nd Pooling Layer
        plt.figure()
        plt.gray()
        pool_2_ = np.array(pool_2_)
        for i in range(60):
            plt.subplot(6, 10, i + 1);
            plt.axis('off');
            plt.imshow(pool_2_[0, :, :, i])

        # Obtain feature maps of convolutional and pooling layers for 2nd test pattern
        sess.run(tf.global_variables_initializer())
        conv_1_out_, pool_1_, conv_2_out_, pool_2_ = sess.run([conv_1_out, pool_1, conv_2_out, pool_2],
                                                              {x: X2.reshape(1, 32 * 32 * 3)})

        # Plot feature map of 1st Convolutional Layer
        plt.figure()
        plt.gray()
        conv_1_out_ = np.array(conv_1_out_)
        for i in range(50):
            plt.subplot(5, 10, i + 1);
            plt.axis('off');
            plt.imshow(conv_1_out_[0, :, :, i])

        # Plot feature map of 1st Pooling Layer
        plt.figure()
        plt.gray()
        pool_1_ = np.array(pool_1_)
        for i in range(50):
            plt.subplot(5, 10, i + 1);
            plt.axis('off');
            plt.imshow(pool_1_[0, :, :, i])

        # Plot feature map of 2nd Convolutional Layer
        plt.figure()
        plt.gray()
        conv_2_out_ = np.array(conv_2_out_)
        for i in range(60):
            plt.subplot(6, 10, i + 1);
            plt.axis('off');
            plt.imshow(conv_2_out_[0, :, :, i])

        # Plot feature map of 2nd Pooling Layer
        plt.figure()
        plt.gray()
        pool_2_ = np.array(pool_2_)
        for i in range(60):
            plt.subplot(6, 10, i + 1);
            plt.axis('off');
            plt.imshow(pool_2_[0, :, :, i])

        plt.show()


if __name__ == '__main__':
    main()