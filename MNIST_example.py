"""
Get familar with MNIST.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)

number_of_features = mnist.train.images.shape[1]
number_of_classes = mnist.train.labels.shape[1]

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([number_of_features, number_of_classes]))
b = tf.Variable(tf.zeros([number_of_classes]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, number_of_classes])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
