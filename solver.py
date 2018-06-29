# ------------------------------------------------------------
# Real-Time Style Transfer Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

import tf_utils as tf_utils
import utils as utils
from style_transfer import StyleTranser, Transfer


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.style_img_name = flags.style_img.split('/')[-1][:-4]
        self.content_target_paths = utils.all_files_under(self.flags.train_path)

        self.test_targets = utils.all_files_under(self.flags.test_path, extension='.jpg')

        self.test_target_names = utils.all_files_under(self.flags.test_path, append_path=False, extension='.jpg')
        self.test_save_paths = [os.path.join(self.flags.test_dir, self.style_img_name, file[:-4])
                                for file in self.test_target_names]

        self.num_contents = len(self.content_target_paths)
        self.num_iters = int(self.num_contents / self.flags.batch_size) * self.flags.epochs

        self.model = StyleTranser(self.sess, self.flags, self.num_iters)

        self.train_writer = tf.summary.FileWriter('logs/{}'.format(self.style_img_name), graph_def=self.sess.graph_def)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        tf_utils.show_all_variables()

    def train(self):
        random.seed(datetime.now())  # set random sedd

        for iter_time in range(self.num_iters):
            # sampling images and save them
            self.sample(iter_time)

            # read batch data and feed forward
            batch_imgs = self.next_batch()
            loss, summary = self.model.train_step(batch_imgs)

            # write log to tensorboard
            self.train_writer.add_summary(summary, iter_time)
            self.train_writer.flush()

            # print loss information
            self.model.print_info(loss, iter_time)

        # save model at the end
        self.save_model()

    def save_model(self):
        model_name = 'model'
        self.saver.save(self.sess, os.path.join(self.flags.checkpoint_dir, self.style_img_name, model_name))
        print('=====================================')
        print('              Model saved!           ')
        print('=====================================\n')

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            self.save_model()  # save model before sample examples

            for idx in range(len(self.test_save_paths)):
                save_path = (self.test_save_paths[idx] + '_%s.png' % iter_time)

                print('save path: {}'.format(save_path))
                print('test_target: {}'.format(self.test_targets[idx]))

                self.feed_transform([self.test_targets[idx]], [save_path])

    def next_batch(self):
        batch_imgs = []
        batch_files = np.random.choice(self.content_target_paths, self.flags.batch_size, replace=False)

        for batch_file in batch_files:
            img = utils.imread(batch_file, img_size=(256, 256, 3))
            batch_imgs.append(img)

        return np.asarray(batch_imgs)

    def feed_transform(self, data_in, paths_out):
        checkpoint_dir = os.path.join(self.flags.checkpoint_dir, self.style_img_name, 'model')
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
            _pred = sess.run(pred, feed_dict={img_placeholder: img})
            utils.imsave(paths_out[0], _pred[0])  # paths_out and _pred is list


