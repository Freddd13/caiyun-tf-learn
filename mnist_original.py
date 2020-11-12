from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# 定义常量
INPUT_NODE = 784
HIDE_NODE = 500
OUTPUT_NODE = 10

REGULARIZER_RATE = 0.0001
LEARNING_RATE = 0.8
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 100
TRAIN_STEPS = 30000

def inference(input, weights1, bias1, weights2, bias2, mavg = False):
    if not mavg:
        print('ignore mavg...')
        hide_output = tf.nn.relu(input @ weights1 + bias1)
        final_output = hide_output @ weights2 + bias2
        return final_output
    else:
        print('use mavg...')
        print("It hasn't been written yet!")

def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None,INPUT_NODE], name='input')
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='output')
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,HIDE_NODE], stddev=0.1))
    bias1 = tf.Variable(tf.constant(0.1, shape=[HIDE_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([HIDE_NODE, OUTPUT_NODE], stddev=0.1))
    bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 前向传播
    y = inference(x, weights1, bias1, weights2, bias2)

    # 反向传播
    ## 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy = tf.reduce_mean(cross_entropy)
    ## 正则化
    regularize = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    regularization = regularize(weights1) + regularize(weights2)
    loss = cross_entropy + regularization
    ## 学习率
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                                               global_step=global_step,
                                               decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                               decay_rate=LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ## 滑动平均添加更新
    ## 准确率定义
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练会话
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x : mnist.validation.images, y_:mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_:mnist.test.labels}

        for i in range(TRAIN_STEPS):
            # 验证
            if i % 100 == 0:
                validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d steps' training, validate_accuracy is %f" % (i, validate_accuracy))
            # 训练
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x:xs, y_:ys})
            # print('loss',sess.run(loss,feed_dict={x:xs, y_:ys}))

        # 测试
        test_accuracy = sess.run(accuracy, feed_dict=test_feed)
        print("After %d steps' training, test_accuracy is %f" % (TRAIN_STEPS, test_accuracy))


def main(argv = None):
    mnist = input_data.read_data_sets("/path/o/MINIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()