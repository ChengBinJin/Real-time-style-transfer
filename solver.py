import os
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

import tf_utils as tf_utils
import utils as utils
from style_transfer import StyleTranser


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.style_img_name = flags.style_img.split('/')[-1][:-4]
        self.content_target_paths = utils.all_files_under(self.flags.train_path)
        self.test_targets = utils.all_files_under(self.flags.test_path)
        self.test_target_names = utils.all_files_under(self.flags.test_path, append_path=False)

        self.num_contents = len(self.content_target_paths)
        self.num_iters = int(self.num_contents / self.flags.batch_size) * self.flags.epochs

        self.model = StyleTranser(self.sess, self.flags, self.num_iters)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        tf_utils.show_all_variables()

    def train(self):
        random.seed(datetime.now())  # set random sedd

        for iter_time in range(self.num_iters+1):
            self.sample(iter_time)

            batch_imgs = self.next_batch()
            loss = self.model.train_step(batch_imgs)

            self.model.print_info(loss, iter_time)
            self.save_model(iter_time)

    def save_model(self, iter_time):
        if np.mod(iter_time, self.flags.save_freq) == 0:
            self.saver.save(self.sess, os.path.join(self.flags.checkpoint_dir, self.style_img_name))
            print('=====================================')
            print('              Model saved!           ')
            print('=====================================\n')

    def sample(self, iter_time):
        # if np.mod(iter_time, self.flags.sample_freq):
        #     for idx in range(len(self.test_targets)):
        #         img_name = self.test_target_names[idx]
        #         print('test image name: {}'.format(img_name))
        #
        #         img = np.asarray([utils.imread(self.test_targets[idx])])
        #         pred_img = self.model.sample_img(img)
        #
        #         file_name = os.path.join(self.flags.test_dir, self.style_img_name,
        #                                  '%s_%s.png'.format(img_name, iter_time))
        #         utils.imsave(file_name, pred_img)
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            img = np.asarray([utils.imread(self.content_target_paths[0], img_size=(256, 256, 3))])
            pred_img = self.model.sample_img(img)
            print('pred_img shape: {}'.format(pred_img.shape))
            file_name = os.path.join(self.flags.test_dir, self.style_img_name, '%s.png' % iter_time)
            print('file_name: {}'.format(file_name))
            utils.imsave(file_name, pred_img[0, :, :, :])

    def next_batch(self):
        batch_imgs = []
        batch_files = np.random.choice(self.content_target_paths, self.flags.batch_size, replace=False)

        for batch_file in batch_files:
            img = utils.imread(batch_file, img_size=(256, 256, 3))
            batch_imgs.append(img)

        return np.asarray(batch_imgs)

    def test(self):
        print('hello solver test!')


