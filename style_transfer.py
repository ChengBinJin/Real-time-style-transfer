import functools
import collections
import tensorflow as tf
import numpy as np
import scipy.io
from operator import mul

import tf_utils as tf_utils
import utils as utils


class StyleTranser(object):
    def __init__(self, sess, flags, num_iters):
        self.sess = sess
        self.flags = flags
        self.num_iters = num_iters

        self.norm = 'instance'
        self.tranfer_train_ops = []

        self.style_target = np.asarray([utils.imread(self.flags.style_img)])  # [H, W, C] -> [1, H, W, C]
        self.style_shape = self.style_target.shape
        self.content_shape = [None, 256, 256, 3]

        self.style_layers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
        # self.style_layers = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')  # original paper
        self.content_layer = 'relu4_2'
        # self.content_layer = 'relu2_2'  # original paper

        self.style_target_gram = {}
        self.content_loss, self.style_loss, self.tv_loss = None, None, None

        self._build_net()

    def _build_net(self):
        # ph: tensorflow placeholder
        self.style_img_ph = tf.placeholder(tf.float32, shape=self.style_shape, name='style_img')
        self.content_img_ph = tf.placeholder(tf.float32, shape=self.content_shape, name='content_img')

        self.transfer = Transfer()
        self.vgg = VGG19(self.flags.vgg_path)

        # step 1: extract style_target feature
        vgg_dic = self.vgg(self.style_img_ph)
        for layer in self.style_layers:
            features = self.sess.run(vgg_dic[layer], feed_dict={self.style_img_ph: self.style_target})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            self.style_target_gram[layer] = gram

        # step 2: extract content_target feature
        content_target_feature = {}
        vgg_content_dic = self.vgg(self.content_img_ph, is_reuse=True)
        content_target_feature[self.content_layer] = vgg_content_dic[self.content_layer]

        # step 3: tranfer content image to predicted image
        self.preds = self.transfer(self.content_img_ph/255.0)
        # step 4: extract vgg feature of the predicted image
        preds_dict = self.vgg(self.preds, is_reuse=True)

        # self.sample_pred = self.transfer(self.sample_img_ph/255.0, is_reuse=True)

        self.content_loss_func(preds_dict, content_target_feature)
        self.style_loss_func(preds_dict)
        self.tv_loss_func(self.preds)

        self.total_loss = self.content_loss + self.style_loss + self.tv_loss
        self.optim = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate).minimize(self.total_loss)

    def content_loss_func(self, preds_dict, content_target_feature):
        # calucate content size and check the feature dimension between content and predicted image
        content_size = self._tensor_size(content_target_feature[self.content_layer]) * self.flags.batch_size
        assert self._tensor_size(content_target_feature[self.content_layer]) == self._tensor_size(
            preds_dict[self.content_layer])

        self.content_loss = self.flags.content_weight * (2 * tf.nn.l2_loss(
            preds_dict[self.content_layer] - content_target_feature[self.content_layer]) / content_size)

    def style_loss_func(self, preds_dict):
        style_losses = []
        for style_layer in self.style_layers:
            layer = preds_dict[style_layer]
            _, height, width, num_filters = map(lambda i: i.value, layer.get_shape())
            feature_size = height * width * num_filters
            feats = tf.reshape(layer, (tf.shape(layer)[0], height * width, num_filters))
            feats_trans = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_trans, feats) / feature_size
            style_gram = self.style_target_gram[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)

        self.style_loss = self.flags.style_weight * functools.reduce(tf.add, style_losses) / self.flags.batch_size

    def tv_loss_func(self, preds):
        # total variation denoising
        tv_y_size = self._tensor_size(preds[:, 1:, :, :])
        tv_x_size = self._tensor_size(preds[:, :, 1:, :])

        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :self.content_shape[1]-1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :self.content_shape[2]-1, :])
        self.tv_loss = self.flags.tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / self.flags.batch_size

    @staticmethod
    def _tensor_size(tensor):
        return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

    def train_step(self, imgs):
        ops = [self.optim, self.content_loss, self.style_loss, self.tv_loss, self.total_loss]
        feed_dict = {self.content_img_ph: imgs}
        _, content_loss, style_loss, tv_loss, total_loss = self.sess.run(ops, feed_dict=feed_dict)

        return [content_loss, style_loss, tv_loss, total_loss]

    def sample_img(self, img):
        return self.sess.run(self.preds, feed_dict={self.content_img_ph: img})

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iter', self.num_iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('content_loss', loss[0]), ('style_loss', loss[1]),
                                                  ('tv_loss', loss[2]), ('total_loss', loss[3]),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)


class Transfer(object):
    def __call__(self, img, name='transfer', is_reuse=False):
        with tf.variable_scope(name, reuse=is_reuse):
            # [H, W, C] -> [H, W, 32]
            conv1 = self._conv_layer(img, num_filters=32, filter_size=9, strides=1, name='conv1')
            # [H, W, 32] -> [H/2, W/2, 64]
            conv2 = self._conv_layer(conv1, num_filters=64, filter_size=3, strides=2, name='conv2')
            # [H/2, W/2, 64] -> [H/4, W/4, 128]
            conv3 = self._conv_layer(conv2, num_filters=128, filter_size=3, strides=2, name='conv3')
            # [H/4, W/4, 128] -> [H/4, W/4, 128]
            resid = self.n_res_blocks(conv3, num_blocks=5, name='res_blocks')
            # [H/4, W/4, 128] -> [H/2, W/2, 64]
            conv_t1 = self._conv_tranpose_layer(resid, num_filters=64, filter_size=3, strides=2, name='trans_conv1')
            # [H/2, W/2, 64] -> [H, W, 32]
            conv_t2 = self._conv_tranpose_layer(conv_t1, num_filters=32, filter_size=3, strides=2, name='trans_conv2')
            # [H, W, 32] -> [H, W, 3]
            conv_t3 = self._conv_layer(conv_t2, num_filters=3, filter_size=9, strides=1, relu=False, name='conv4')
            preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2

            return preds

    @staticmethod
    def _conv_layer(input_, num_filters=32, filter_size=3, strides=1, relu=True, name=None):
        with tf.variable_scope(name):
            input_ = tf_utils.conv2d(input_, output_dim=num_filters, k_h=filter_size, k_w=filter_size,
                                     d_h=strides, d_w=strides)
            input_ = tf_utils.instance_norm(input_)

            if relu:
                input_ = tf.nn.relu(input_)

        return input_

    @staticmethod
    def n_res_blocks(x, _ops=None, norm_='instance', is_train=True, num_blocks=6, is_print=False, name=None):
        with tf.variable_scope(name):
            output = None
            for idx in range(1, num_blocks + 1):
                output = tf_utils.res_block(x, x.get_shape()[3], _ops=_ops, norm_=norm_, is_train=is_train,
                                            name='res{}'.format(idx))
                x = output

            if is_print:
                tf_utils.print_activations(output)

        return output

    @staticmethod
    def _conv_tranpose_layer(input_, num_filters=32, filter_size=3, strides=2, name=None):
        with tf.variable_scope(name):
            input_ = tf_utils.deconv2d(input_, num_filters, k_h=filter_size, k_w=filter_size, d_h=strides, d_w=strides)
            input_ = tf_utils.instance_norm(input_)

        return tf.nn.relu(input_)


class VGG19(object):
    def __init__(self, data_path):
        self.data = scipy.io.loadmat(data_path)
        self.weights = self.data['layers'][0]

        self.mean_pixel = np.asarray([123.68, 116.779, 103.939])
        self.layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
        )

    def __call__(self, img, name='vgg', is_reuse=False):
        with tf.variable_scope(name, reuse=is_reuse):
            img_pre = self.preprocess(img)

            net_dic = {}
            current = img_pre
            for i, name in enumerate(self.layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = self.weights[i][0][0][0][0]
                    # matconvent: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = bias.reshape(-1)
                    current = self._conv_layer(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = self._pool_layer(current)

                net_dic[name] = current

            assert len(net_dic) == len(self.layers)

        return net_dic

    @staticmethod
    def _conv_layer(input_, weights, bias):
        conv = tf.nn.conv2d(input_, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
        return tf.nn.bias_add(conv, bias)

    @staticmethod
    def _pool_layer(input_):
        return tf.nn.max_pool(input_, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    def preprocess(self, img):
        return img - self.mean_pixel

    def unprocess(self, img):
        return img + self.mean_pixel
