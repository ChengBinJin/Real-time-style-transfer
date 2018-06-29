# ------------------------------------------------------------
# Real-Time Style Transfer Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from Logan Engstrom
# Email: sbkim0407@gmail.com
# ------------------------------------------------------------
import os
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict

from style_transfer import Transfer
import utils as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/la_muse',
                       'dir to read checkpoint in, default: ./checkpoints/la_muse')

tf.flags.DEFINE_string('in_path', './examples/test', 'test image path, default: ./examples/test')
tf.flags.DEFINE_string('out_path', './examples/results',
                       'destination dir of transformed files, default: ./examples/restuls')


def feed_transform(data_in, paths_out, checkpoint_dir):
    img_shape = utils.imread(data_in[0]).shape

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), tf.Session(config=soft_config) as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=[None, *img_shape], name='img_placeholder')

        model = Transfer()
        pred = model(img_placeholder)

        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception('No checkpoint found...')
        else:
            saver.restore(sess, checkpoint_dir)

        img = np.asarray([utils.imread(data_in[0])]).astype(np.float32)
        start_tic = time.time()
        _pred = sess.run(pred, feed_dict={img_placeholder: img})
        end_toc = time.time()
        print('PT: {:.2f} msec.\n'.format((end_toc - start_tic) * 1000))
        utils.imsave(paths_out[0], _pred[0])  # paths_out and _pred is list


def feed_forward(in_paths, out_paths, checkpoint_dir):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)

    for idx in range(len(in_paths)):
        in_image = in_paths[idx]
        out_image = out_paths[idx]

        shape = "%dx%dx%d" % utils.imread(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)

    for shape in in_path_of_shape:
        print('Processing images of shape {}'.format(shape))
        feed_transform(in_path_of_shape[shape], out_path_of_shape[shape], checkpoint_dir)


def check_opts(flags):
    utils.exists(flags.checkpoint_dir, 'checkpoint_dir not found!')
    utils.exists(flags.in_path, 'in_path not found!')

    style_name = FLAGS.checkpoint_dir.split('/')[-1]
    if not os.path.isdir(os.path.join(flags.out_path, style_name)):
        os.makedirs(os.path.join(flags.out_path, style_name))


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
    check_opts(FLAGS)

    style_name = FLAGS.checkpoint_dir.split('/')[-1]
    img_paths = utils.all_files_under(FLAGS.in_path)
    out_paths = [os.path.join(FLAGS.out_path, style_name, file)
                 for file in utils.all_files_under(FLAGS.in_path, append_path=False)]

    feed_forward(img_paths, out_paths, FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()
