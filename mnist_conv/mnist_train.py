import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

from mnist_fc import mnist_inference

BATCH_SIZE = 100
TRAIN_STEPS = 10000
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZE_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99

SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None,
                                          mnist_inference.INPUT_NODE,
                                          mnist_inference.INPUT_NODE,
                                          mnist_inference.INPUT_CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, shape = [None, mnist_inference.OUTPUT_NODE], name='y-input')
    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZE_RATE)
    # 前向传播，同时加入正则化的损失
    y = mnist_inference.inference(x, regularizer)
    # 指数衰减学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    average_op = ema.apply(tf.trainable_variables())
    # 损失函数，并加和两部分为train_op操作
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, average_op]):
        train_op = tf.no_op(name='train')

    # 初始化saver
    saver = tf.train.Saver()
    # 训练sess
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for step in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs = np.reshape(xs, (BATCH_SIZE,
                                 mnist_inference.INPUT_NODE,
                                 mnist_inference.INPUT_NODE,
                                 mnist_inference.INPUT_CHANNEL))
            _, loss_value, step_value = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})

            if step % 1000 == 0:
                # 每1000轮保存一次模型
                saver.save(sess, os.path.join(SAVE_PATH, MODEL_NAME),global_step=global_step)
                print("===== Step %d : loss is %f, model saved. =====" % (step_value, loss_value))
            print("Step %d: loss is %f." % (step_value, loss_value))
        print("After %d steps' training, loss is %f" % (TRAIN_STEPS, sess.run(loss, feed_dict={x:xs, y_:ys})))

def main(argv = None):
    mnist = input_data.read_data_sets('/path/to/mnist_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()



