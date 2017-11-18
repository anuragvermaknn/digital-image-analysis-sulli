# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import tensorflow as tf
from PIL import Image
import numpy as np

from datasets import dataset_utils
import tensorflow.contrib.slim.python.slim.data.dataset_data_provider as dataset_data_provider

from properties import training_params
from properties import disk_storage as disk_storage_props
slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

_FILE_PATTERN = 'lymph_nodes_%s_*.tfrecord' #'flowers_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': training_params.PATCHES_TRAIN_SAMPLES,
                   'validation': training_params.PATCHES_VALIDATION_SAMPLES}

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}



def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)



def main(unused_argv):
    # if platform.system() == 'Linux':
    #   dataset_dir = "/home/anurag/Desktop/tensorflow_data"
    # else:
    #   dataset_dir = "/Users/anuragverma/Desktop/tensorflow_data"

    dataset_dir = disk_storage_props.PATCHES_TRAIN_DATA_DIR
    tf_record_dir = disk_storage_props.PATCHES_TF_RECORD_DIR

    dataset = get_split('train', tf_record_dir)
    dataset_provider = dataset_data_provider.DatasetDataProvider(dataset,
                                                                 num_readers=12,
                                                                 shuffle=False
                                                                 )

    with tf.Session() as sess:
        #dataset_dir = "/Users/anuragverma/Desktop/tensorflow_data"
        tf.train.start_queue_runners()
        [image_raw, label] = dataset_provider.get(['image', 'label'])
        #image_decoded = tf.image.decode_jpeg(image_raw)
        image_raw, label = sess.run([image_raw, label])
        print(label)
        print(image_raw.shape)
        #Image.fromarray(np.array(image_raw)).show()
        #print(image_decoded.shape)
        #print(image_decoded)
        #print("\n\n\n\n",image_raw)
        print (dataset_provider.num_samples())
        #images, labels = dataset_provider.get(['image', 'label'])
        for i in range(dataset_provider.num_samples()):
            [image_raw, label] = dataset_provider.get(['image','label'])
            image_raw, label = sess.run([image_raw, label])
            if label == 0:
                Image.fromarray(np.array(image_raw)).show()

    #print([labels])
    #print(type(images))

if __name__ == "__main__":
    tf.app.run()