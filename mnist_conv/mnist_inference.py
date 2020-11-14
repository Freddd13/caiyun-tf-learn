import tensorflow as tf


INPUT_NODE = 784
LAYER1_NODE = 500
OUTPUT_NODE = 10

def get_weights_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        print('add to collection')
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weights_variable([INPUT_NODE, LAYER1_NODE], regularizer=regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
    layer1 = tf.nn.relu(input_tensor @ weights + biases)

    with tf.variable_scope('layer2'):
        weights = get_weights_variable([LAYER1_NODE, OUTPUT_NODE], regularizer=regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
    layer2 = layer1 @ weights + biases

    return layer2

