import tensorflow as tf
from mnist_fc import mnist_train
from mnist_fc import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data
import time

EVALUATE_INTERVAL = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validation_feed = {x:mnist.validation.images, y_:mnist.validation.labels}

        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 变量重命名
        ema = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        v_restore = ema.variables_to_restore()
        saver = tf.train.Saver(v_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # global_step
                    accuracy_score = sess.run(accuracy, feed_dict=validation_feed)
                    print('accuracy score is %f .' % accuracy_score)
                else:
                    print('No checkpoint is found.')
            time.sleep(EVALUATE_INTERVAL)


def main(argv = None):
    mnist = input_data.read_data_sets('/path/to/mnist_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()

