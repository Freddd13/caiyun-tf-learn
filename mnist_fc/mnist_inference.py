import tensorflow as tf


INPUT_CHANNEL = 1
INPUT_NODE = 28
CONV1_DEEP = 32
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
FC_SIZE = 512
OUTPUT_NODE = 10


def get_weights_variable(shape, regularizer):
    print('-'*50 + '\n',shape)
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        print('add to collection')
        tf.add_to_collection('losses', regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weight = tf.get_variable("weights", [CONV1_SIZE, CONV1_SIZE, INPUT_CHANNEL, CONV1_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.1))
    conv1 = tf.nn.conv2d(input_tensor, conv1_weight, [1,1,1,1], padding="SAME")
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))

    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, [1,2,2,1],[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weight = tf.get_variable("weights", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.conv2d(pool1, conv2_weight, [1,1,1,1], padding="SAME")
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, [1,2,2,1], [1,2,2,1], padding="SAME")

    # reshape pool2 for the final layer
    # pool2_shape = pool2.get_shape().as_list()
    pool2_shape = tf.shape(pool2)
    fc1_input_nodes = [pool2_shape[0], pool2_shape[1] * pool2_shape[2] * pool2_shape[3]]
    # print(type(fc1_input_nodes[1]))
    fc1_input = tf.reshape(pool2, fc1_input_nodes)
    tmp = pool2.get_shape().as_list()
    insize = tmp[1]*tmp[2]*tmp[3]


    with tf.variable_scope("layer5-fc1"):
        fc1_weight = get_weights_variable([insize, FC_SIZE], regularizer)
        fc1_bias = tf.get_variable("bias", [FC_SIZE],
                                   initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(fc1_input @ fc1_weight + fc1_bias)
        # if train====
        fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("layer6-fc2"):
        fc2_weight = get_weights_variable([FC_SIZE, OUTPUT_NODE], regularizer)
        fc2_bias = tf.get_variable("bias", [OUTPUT_NODE],
                                   initializer=tf.constant_initializer(0.1))
        logit = fc1 @ fc2_weight + fc2_bias

    return logit
