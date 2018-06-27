# ---------------------------------------------------------
# Tensorflow Utils Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# ---------------------------------------------------------
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.training import moving_averages


def padding2d(x, p_h=1, p_w=1, pad_type='REFLECT', name='pad2d'):
    if pad_type == 'REFLECT':
        return tf.pad(x, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], 'REFLECT', name=name)


def conv2d(x, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME', name='conv2d', is_print=True):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        if is_print:
            print_activations(conv)

        return conv


def deconv2d(x, k, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, padding_='SAME', output_size=None,
             name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        input_shape = x.get_shape().as_list()

        # calculate output size
        h_output, w_output = None, None
        if not output_size:
            h_output, w_output = input_shape[1] * 2, input_shape[2] * 2
        # output_shape = [input_shape[0], h_output, w_output, k]  # error when not define batch_size
        output_shape = [tf.shape(x)[0], h_output, w_output, k]

        # conv2d transpose
        w = tf.get_variable('w', [k_h, k_w, k, input_shape[3]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, d_h, d_w, 1],
                                        padding=padding_)

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def upsampling2d(x, size=(2, 2), name='upsampling2d'):
    with tf.name_scope(name):
        shape = x.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(x, size=(size[0] * shape[1], size[1] * shape[2]))


def linear(x, output_size, bias_start=0.0, with_w=False, name='fc'):
    shape = x.get_shape().as_list()
    # print('shape: ', shape)

    with tf.variable_scope(name):
        matrix = tf.get_variable(name="matrix", shape=[shape[1], output_size],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name="bias", shape=[output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def norm(x, name, _type, _ops, is_train=True):
    if _type == 'batch':
        return batch_norm(x, name=name, _ops=_ops, is_train=is_train)
    elif _type == 'instance':
        return instance_norm(x, name=name)
    else:
        raise NotImplementedError


def batch_norm(x, name, _ops, is_train=True):
    """Batch normalization."""
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable('beta', params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        if is_train is True:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                          initializer=tf.constant_initializer(0.0, tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)

            _ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
            _ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
            variance = tf.get_variable('moving_variance', params_shape, tf.float32, trainable=False)

        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)
        y.set_shape(x.get_shape())

        return y


def instance_norm(x, name='instance_norm', mean=1.0, stddev=0.02, epsilon=1e-5):
    with tf.variable_scope(name):
        depth = x.get_shape()[3]
        scale = tf.get_variable(
            'scale', [depth], tf.float32,
            initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))

        # calcualte mean and variance as instance
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)

        # normalization
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv

        return scale * normalized + offset


def n_res_blocks(x, _ops=None, norm_='instance', is_train=True, num_blocks=6, is_print=False):
    output = None
    for idx in range(1, num_blocks+1):
        output = res_block(x, x.get_shape()[3], _ops=_ops, norm_=norm_, is_train=is_train,
                           name='res{}'.format(idx))
        x = output

    if is_print:
        print_activations(output)

    return output


# norm(x, name, _type, _ops, is_train=True)
def res_block(x, k, _ops=None, norm_='instance', is_train=True, pad_type=None, name=None):
    with tf.variable_scope(name):
        conv1, conv2 = None, None

        # 3x3 Conv-Batch-Relu S1
        with tf.variable_scope('layer1'):
            if pad_type is None:
                conv1 = conv2d(x, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv')
            elif pad_type == 'REFLECT':
                padded1 = padding2d(x, p_h=1, p_w=1, pad_type='REFLECT', name='padding')
                conv1 = conv2d(padded1, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID', name='conv')
            normalized1 = norm(conv1, name='norm', _type=norm_, _ops=_ops, is_train=is_train)
            relu1 = tf.nn.relu(normalized1)

        # 3x3 Conv-Batch S1
        with tf.variable_scope('layer2'):
            if pad_type is None:
                conv2 = conv2d(relu1, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv')
            elif pad_type == 'REFLECT':
                padded2 = padding2d(relu1, p_h=1, p_w=1, pad_type='REFLECT', name='padding')
                conv2 = conv2d(padded2, k, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID', name='conv')
            normalized2 = norm(conv2, name='norm', _type=norm_, _ops=_ops, is_train=is_train)

    # sum layer1 and layer2
    output = x + normalized2
    return output


def identity(x, name='identity', is_print=False):
    output = tf.identity(x, name=name)
    if is_print:
        print_activations(output)

    return output


def max_pool_2x2(x, name='max_pool'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def sigmoid(x, name='sigmoid', is_print=False):
    output = tf.nn.sigmoid(x, name=name)
    if is_print:
        print_activations(output)

    return output


def tanh(x, name='tanh', is_print=False):
    output = tf.nn.tanh(x, name=name)
    if is_print:
        print_activations(output)

    return output


def relu(x, name='relu', is_print=False):
    output = tf.nn.relu(x, name=name)
    if is_print:
        print_activations(output)

    return output


def lrelu(x, leak=0.2, name='lrelu', is_print=False):
    output = tf.maximum(x, leak*x, name=name)
    if is_print:
        print_activations(output)

    return output


def xavier_init(in_dim):
    print('in_dim: ', in_dim)
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return xavier_stddev


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def batch_convert2int(images):
    # images: 4D float tensor (batch_size, image_size, image_size, depth)
    return tf.map_fn(convert2int, images, dtype=tf.uint8)


def convert2int(image):
    # transform from float tensor ([-1.,1.]) to int image ([0,255])
    return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)
