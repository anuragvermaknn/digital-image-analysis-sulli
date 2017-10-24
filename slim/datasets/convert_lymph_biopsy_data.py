# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Downloads and converts MNIST data to TFRecords of TF-Example protos.

This module downloads the MNIST data, uncompresses it, reads the files
that make up the MNIST data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

import properties.disk_storage as disk_storage_utils


flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir',disk_storage_utils.LYMPH_DATA_DIR , 'String: Your dataset directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.3, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', 'wsi_%s_%05d-of-%05d.tfrecord', 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS


class ImageReader(object):


    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)


    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def png_to_jpeg(self, sess, image_data):
        return sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  return (filename.endswith('.png') or filename.endswith('.PNG'))


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  flower_root = os.path. join(dataset_dir)
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = FLAGS.tfrecord_filename % (
      split_name, shard_id, FLAGS.num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(FLAGS.num_shards)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(FLAGS.num_shards):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()

            # Convert any PNG to JPEG's for consistency.
            if _is_png(filenames[i]):
                print('Converting PNG to JPEG for %s' % filenames[i])
                image_data = image_reader.png_to_jpeg(sess, image_data)

            # Decode the RGB JPEG.
            #image_data = image_reader.decode_jpeg(sess, image_read)

            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


# def _clean_up_temporary_files(dataset_dir):
#   """Removes temporary files used to create the dataset.
#
#   Args:
#     dataset_dir: The directory where the temporary files are stored.
#   """
#   filename = _DATA_URL.split('/')[-1]
#   filepath = os.path.join(dataset_dir, filename)
#   tf.gfile.Remove(filepath)
#
#   tmp_dir = os.path.join(dataset_dir, 'flower_photos')
#   tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(FLAGS.num_shards):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True




def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
    dataset_dir: The dataset directory where the dataset is stored.
    """

    # ==============================================================CHECKS==========================================================================

    # Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
      raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    # Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
      raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    # If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir=FLAGS.dataset_dir):
      print('Dataset files already exist. Exiting without re-creating them.')
      return None
    # ==============================================================END OF CHECKS===================================================================

    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    # Divide into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]
    # training_filenames = photo_filenames[_NUM_VALIDATION:]
    # validation_filenames = photo_filenames[:_NUM_VALIDATION]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    #_clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Lymph WSI dataset!')


def main(unused_argv):
    if not (os.path.isdir(FLAGS.dataset_dir)):
        os.makedirs(FLAGS.dataset_dir)

    run(FLAGS.dataset_dir)


if __name__ == '__main__':
  tf.app.run()

