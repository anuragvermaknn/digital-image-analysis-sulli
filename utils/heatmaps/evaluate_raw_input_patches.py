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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import glob, time
import cv2, os
from PIL import Image

import glob, math
import cv2, os
import numpy as np

from datasets import raw_patches_for_heatmaps
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from properties import wsi_props, training_params
from properties import disk_storage as disk_storage_props
from utils.heatmaps.consecutive_patch_utils import read_patches_count_file
from utils.heatmaps import compute_heatmap_utils
slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', disk_storage_props.TRAIN_LOGS_DIR,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', disk_storage_props.EVAL_LOGS_DIR, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

# tf.app.flags.DEFINE_string(
#     'dataset_name', wsi_props.DATASET_NAME, 'The name of the dataset to load.')
#
# tf.app.flags.DEFINE_string(
#     'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', disk_storage_props.PATCHES_TF_RECORD_DIR, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', wsi_props.PATCH_SIZE, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

tumor_predicted = 0
non_tumor_predicted = 0

# def main(_):
#
#   if not FLAGS.dataset_dir:
#     raise ValueError('You must supply the dataset directory with --dataset_dir')
#
#   tf.logging.set_verbosity(tf.logging.INFO)
#   with tf.Graph().as_default() and tf.Session() as sess:
#
#     tf.train.start_queue_runners()
#     tf_global_step = slim.get_or_create_global_step()
#
#     ######################
#     # Select the dataset #
#     ######################
#     # dataset = dataset_factory.get_dataset(
#     #     FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
#
#     # wsi_name = "Normal_001"
#     # normal_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_NORMAL_DATA_DIR, '*.tif'))
#     # normal_wsi_paths.sort()
#     # wsi_path = normal_wsi_paths[0]
#     mask_image_resolution_level = wsi_props.MASK_IMAGE_RESOLUTION_LEVEL
#
#     wsi_name = "Tumor_001"
#     tumor_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_TUMOR_DATA_DIR, '*.tif'))
#     tumor_wsi_paths.sort()
#     wsi_path = tumor_wsi_paths[0]
#
#     dataset_dir = disk_storage_props.WSI_RAW_PATCHES_PARENT_DIR_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
#     tf_record_dir = disk_storage_props.WSI_RAW_PATCHES_TF_RECORD_DIR_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
#     patches_count = read_patches_count_file(wsi_name=wsi_name)
#     dataset = raw_patches_for_heatmaps.get_split('eval', tf_record_dir, patches_count)
#     # dataset_provider = dataset_data_provider.DatasetDataProvider(dataset,
#     #                                                              num_readers=12,
#     #                                                              shuffle=False
#     #                                                              )
#
#
#     heatmap_probabilities_array = compute_heatmap_utils.\
#         get_default_heatmap_probabilities_array(wsi_path, mask_image_resolution_level=mask_image_resolution_level)
#
#     ####################
#     # Select the model #
#     ####################
#     network_fn = nets_factory.get_network_fn(
#         FLAGS.model_name,
#         num_classes=(dataset.num_classes - FLAGS.labels_offset),
#         is_training=False)
#
#     ##############################################################
#     # Create a dataset provider that loads data from the dataset #
#     ##############################################################
#     provider = slim.dataset_data_provider.DatasetDataProvider(
#         dataset,
#         shuffle=False,
#         common_queue_capacity=2 * FLAGS.batch_size,
#         common_queue_min=FLAGS.batch_size)
#     [image, label, filename] = provider.get(['image', 'label', 'filename'])
#     label -= FLAGS.labels_offset
#
#     #####################################
#     # Select the preprocessing function #
#     #####################################
#     preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
#     image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#         preprocessing_name,
#         is_training=False)
#
#     eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
#
#     image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
#
#     images, labels, filenames = tf.train.batch(
#         [image, label, filename],
#         batch_size=FLAGS.batch_size,
#         num_threads=FLAGS.num_preprocessing_threads,
#         capacity=5 * FLAGS.batch_size)
#
#     ####################
#     # Define the model #
#     ####################
#     #logits, _ = network_fn(images)
#     logits, end_points = network_fn(images)
#
#     if FLAGS.moving_average_decay:
#       variable_averages = tf.train.ExponentialMovingAverage(
#           FLAGS.moving_average_decay, tf_global_step)
#       variables_to_restore = variable_averages.variables_to_restore(
#           slim.get_model_variables())
#       variables_to_restore[tf_global_step.op.name] = tf_global_step
#     else:
#       variables_to_restore = slim.get_variables_to_restore()
#
#     predictions = tf.argmax(logits, 1)
#     labels = tf.squeeze(labels)
#
#     # Define the metrics:
#     # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
#     #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
#     #     'Recall_5': slim.metrics.streaming_recall_at_k(
#     #         logits, labels, 5),
#     # })
#
#     # Print the summaries to screen.
#     # for name, value in names_to_values.items():
#     #   summary_name = 'eval/%s' % name
#     #   op = tf.summary.scalar(summary_name, value, collections=[])
#     #   op = tf.Print(op, [value], summary_name)
#     #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
#
#     # TODO(sguada) use num_epochs=1
#     if FLAGS.max_num_batches:
#       num_batches = FLAGS.max_num_batches
#     else:
#       # This ensures that we make a single pass over all of the data.
#       print("dataset.num_samples : ", dataset.num_samples)
#       num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
#       print("num_batches : ", num_batches)
#
#     if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
#       checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
#       print ("checkpoint_path\n", checkpoint_path)
#     else:
#       checkpoint_path = FLAGS.checkpoint_path
#
#     tf.logging.info('Evaluating %s' % checkpoint_path)
#
#     tf.train.start_queue_runners()
#     print("getting variables to restore ")
#     tf_saver = tf.train.Saver(variables_to_restore)
#     print("Starting to restore model")
#     tf_saver.restore(sess, checkpoint_path)
#     print("Model restored")
#
#     batches_inferred = 0
#     while batches_inferred < num_batches:
#         predictions_batch, filenames_batch, end_points_batch = sess.run([predictions, filenames, end_points])
#         #print (predictions_batch, "\n", filenames_batch)
#         batches_inferred += 1
#         #print ("\n\n\n")
#         #print ("end_points \n", end_points_batch['Predictions'])
#
#         heatmap_probabilities_array = compute_heatmap_utils.aggregate_heatmap_probabilities(wsi_name=wsi_name,
#                                                                                             heatmap_probabilities_array=heatmap_probabilities_array,
#                                                                                             end_point_predictions_batch=end_points_batch['Predictions'],
#                                                                                             filenames_batch=filenames_batch)
#
#         compute_heatmap_utils.save_filenames_and_predictions_for_review(wsi_name=wsi_name,
#                                                                         filenames_batch=filenames_batch,
#                                                                         predictions_batch=predictions_batch)
#
#         aggregate_predictions(predictions_batch=predictions_batch)
#
#     print("tumor_predicted :", tumor_predicted, ", non_tumor_predicted : ", non_tumor_predicted)
#     heatmap_filepath = disk_storage_props.WSI_HEATMAP_OUPUT_FILE.replace("WSI_NAME", wsi_name)
#     print( "heatmap_filepath ",heatmap_filepath)
#     # Image.fromarray(heatmap_probabilities_array*255).save(fp=heatmap_filepath,
#     #                                                       format="PNG")
#     if not (os.path.isdir(disk_storage_props.HEATMAP_OUTPUT_DATA_DIR)):
#         os.makedirs(disk_storage_props.HEATMAP_OUTPUT_DATA_DIR)
#     print("heatmap_probabilities_array*255\n", heatmap_probabilities_array*255)
#     cv2.imwrite(heatmap_filepath, heatmap_probabilities_array*255)
#     print(" file saved at ", heatmap_filepath)
#     # slim.evaluation.evaluate_once(
#     #     master=FLAGS.master,
#     #     checkpoint_path=checkpoint_path,
#     #     logdir=FLAGS.eval_dir,
#     #     num_evals=num_batches,
#     #     eval_op=list(names_to_updates.values()),
#     #     variables_to_restore=variables_to_restore)


def aggregate_predictions(predictions_batch):
    global tumor_predicted, non_tumor_predicted
    for prediction in predictions_batch:
        if prediction == 0:
            non_tumor_predicted += 1
        else:
            tumor_predicted += 1

def evaluate_for_a_wsi(wsi_name,
                       wsi_path,
                       wsi_mask_name=None,
                       wsi_mask_path=None):

  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() and tf.Session() as sess:    

    tf.train.start_queue_runners()
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    # dataset = dataset_factory.get_dataset(
    #     FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    # wsi_name = "Normal_001"
    # normal_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_NORMAL_DATA_DIR, '*.tif'))
    # normal_wsi_paths.sort()
    # wsi_path = normal_wsi_paths[0]
    mask_image_resolution_level = wsi_props.MASK_IMAGE_RESOLUTION_LEVEL

    # wsi_name = "Tumor_001"
    # tumor_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_TUMOR_DATA_DIR, '*.tif'))
    # tumor_wsi_paths.sort()
    # wsi_path = tumor_wsi_paths[0]

    dataset_dir = disk_storage_props.WSI_RAW_PATCHES_PARENT_DIR_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
    tf_record_dir = disk_storage_props.WSI_RAW_PATCHES_TF_RECORD_DIR_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
    patches_count = read_patches_count_file(wsi_name=wsi_name)
    dataset = raw_patches_for_heatmaps.get_split('eval', tf_record_dir, patches_count)
    # dataset_provider = dataset_data_provider.DatasetDataProvider(dataset,
    #                                                              num_readers=12,
    #                                                              shuffle=False
    #                                                              )


    heatmap_probabilities_array = compute_heatmap_utils.\
        get_default_heatmap_probabilities_array(wsi_path, mask_image_resolution_level=mask_image_resolution_level)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label, filename] = provider.get(['image', 'label', 'filename'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels, filenames = tf.train.batch(
        [image, label, filename],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    #logits, _ = network_fn(images)
    logits, end_points = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    #     'Recall_5': slim.metrics.streaming_recall_at_k(
    #         logits, labels, 5),
    # })

    # Print the summaries to screen.
    # for name, value in names_to_values.items():
    #   summary_name = 'eval/%s' % name
    #   op = tf.summary.scalar(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      print("dataset.num_samples : ", dataset.num_samples)
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
      print("num_batches : ", num_batches)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
      print ("checkpoint_path\n", checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    tf.train.start_queue_runners()
    print("getting variables to restore ")
    tf_saver = tf.train.Saver(variables_to_restore)
    print("Starting to restore model")
    tf_saver.restore(sess, checkpoint_path)
    print("Model restored")

    batches_inferred = 0
    while batches_inferred < num_batches:
        predictions_batch, filenames_batch, end_points_batch = sess.run([predictions, filenames, end_points])
        #print (predictions_batch, "\n", filenames_batch)
        batches_inferred += 1
        #print ("\n\n\n")
        #print ("end_points \n", end_points_batch['Predictions'])

        heatmap_probabilities_array = compute_heatmap_utils.aggregate_heatmap_probabilities(wsi_name=wsi_name,
                                                                                            heatmap_probabilities_array=heatmap_probabilities_array,
                                                                                            end_point_predictions_batch=end_points_batch['Predictions'],
                                                                                            filenames_batch=filenames_batch)

        compute_heatmap_utils.save_filenames_and_predictions_for_review(wsi_name=wsi_name,
                                                                        filenames_batch=filenames_batch,
                                                                        predictions_batch=predictions_batch)

        aggregate_predictions(predictions_batch=predictions_batch)

    print("tumor_predicted :", tumor_predicted, ", non_tumor_predicted : ", non_tumor_predicted)
    heatmap_filepath = disk_storage_props.WSI_HEATMAP_OUPUT_FILE.replace("WSI_NAME", wsi_name)
    print( "heatmap_filepath ",heatmap_filepath)
    # Image.fromarray(heatmap_probabilities_array*255).save(fp=heatmap_filepath,
    #                                                       format="PNG")
    if not (os.path.isdir(disk_storage_props.HEATMAP_OUTPUT_DATA_DIR)):
        os.makedirs(disk_storage_props.HEATMAP_OUTPUT_DATA_DIR)
    print("heatmap_probabilities_array*255\n", heatmap_probabilities_array*255)
    cv2.imwrite(heatmap_filepath, heatmap_probabilities_array*255)
    print(" file saved at ", heatmap_filepath)

    # clean noise in heatmap using morphology images
    cleaned_1, cleaned_2 = compute_heatmap_utils.clean_heatmap_using_morph_ops(heatmap_filepath=heatmap_filepath)
    heatmap_cleaned_1_filepath = disk_storage_props.WSI_HEATMAP_CLEANED_FILE_1.replace("WSI_NAME", wsi_name)
    heatmap_cleaned_2_filepath = disk_storage_props.WSI_HEATMAP_CLEANED_FILE_2.replace("WSI_NAME", wsi_name)

    heatmap_probabilities_array_image = np.array(heatmap_probabilities_array)*255
    #print(heatmap_probabilities_array_image)
    if wsi_mask_name is not None:
        heatmap_with_actual_mask = compute_heatmap_utils.\
            annotate_heatmap_probabilities_array_with_original_mask(
            heatmap_probabilities_array_image=heatmap_probabilities_array,
            wsi_mask_path=wsi_mask_path,
            mask_image_resolution_level=mask_image_resolution_level)
        x = np.concatenate((heatmap_probabilities_array*255, heatmap_with_actual_mask), axis=1)
        #y = np.concatenate((heatmap_probabilities_array*255, heatmap_with_actual_mask ), axis=1)
        heatmap_with_actual_mask_filepath = disk_storage_props.WSI_HEATMAP_WITH_ACTUAL_MASK_OUPUT_FILE.replace("WSI_NAME", wsi_name)
        cv2.imwrite(heatmap_with_actual_mask_filepath, x)

        # add mask to cleaned heatmaps also for review
        cleaned_1 = np.concatenate((cleaned_1, heatmap_with_actual_mask), axis=1)
        cleaned_2 = np.concatenate((cleaned_2, heatmap_with_actual_mask), axis=1)

    cv2.imwrite(heatmap_cleaned_1_filepath, cleaned_1)
    cv2.imwrite(heatmap_cleaned_2_filepath, cleaned_2)

        # slim.evaluation.evaluate_once(
    #     master=FLAGS.master,
    #     checkpoint_path=checkpoint_path,
    #     logdir=FLAGS.eval_dir,
    #     num_evals=num_batches,
    #     eval_op=list(names_to_updates.values()),
    #     variables_to_restore=variables_to_restore)
  tf.reset_default_graph()


def main(_):
    tumor_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_TUMOR_DATA_DIR, '*.tif'))
    tumor_wsi_paths.sort()

    tumor_mask_paths = glob.glob(os.path.join(disk_storage_props.RAW_TUMOR_MASK_DIR, '*.tif'))
    tumor_mask_paths.sort()
    tumor_image_mask_pairs = list(zip(tumor_wsi_paths, tumor_mask_paths))

    start_time = time.time()
    step = 0
    for wsi_path, wsi_mask_path in tumor_image_mask_pairs:
        step += 1
        if step == 1:
            continue
        wsi_name = wsi_path.split('/')[-1].split('.')[0]
        wsi_mask_name = wsi_mask_path.split('/')[-1].split('.')[0]
        
        evaluate_for_a_wsi(wsi_name=wsi_name, wsi_path=wsi_path, wsi_mask_name=wsi_mask_name, wsi_mask_path=wsi_mask_path)
        duration = time.time() - start_time
        print( 'For ', wsi_name, '\t',' evaluated input patches to tf records : %d minutes' % math.ceil(duration/60))
        start_time = time.time()
        if step >= 11:
            break
        

if __name__ == '__main__':
  tf.app.run()
