# ------------------------------------------------------------
# Real-Time Style Transfer Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin, based on code from Logan Engstrom
# Email: sbkim0407@gmail.com
# ------------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
from moviepy.video.io.VideoFileClip import VideoFileClip

import utils as utils
from style_transfer import Transfer

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('checkpoint_dir', 'checkpoints/la_muse',
                       'dir to read checkpoint in, default: ./checkpoints/la_muse')
tf.flags.DEFINE_string('in_path', None, 'input video path')
tf.flags.DEFINE_string('out_path', None, 'path to save processeced video to')


def feed_forward_video(path_in, path_out, checkpoint_dir):
    # initialize video cap
    video_cap = VideoFileClip(path_in, audio=False)
    # initialize writer
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_cap.size, video_cap.fps, codec='libx264',
                                                    preset='medium', bitrate='2000k', audiofile=path_in,
                                                    threads=None, ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), tf.Session(config=soft_config) as sess:
        batch_shape = (None, video_cap.size[1], video_cap.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

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

        frame_id = 0
        for frame in video_cap.iter_frames():
            print('frame id: {}'.format(frame_id))
            _pred = sess.run(pred, feed_dict={img_placeholder: np.asarray([frame]).astype(np.float32)})
            video_writer.write_frame(np.clip(_pred, 0, 255).astype(np.uint8))
            frame_id += 1

        video_writer.close()


def check_opts(flags):
    utils.exists(flags.checkpoint_dir, 'checkpoint_dir not found!')
    utils.exists(flags.in_path, 'in_path not found!')


def main(_):
    os.environ['CUDA_AVAILABLE_DEVICES'] = FLAGS.gpu_index
    check_opts(FLAGS)

    feed_forward_video(FLAGS.in_path, FLAGS.out_path, FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()
