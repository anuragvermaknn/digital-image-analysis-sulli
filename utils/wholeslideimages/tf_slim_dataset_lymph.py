import tensorflow as tf
from datasets import lymph_biopsy
from properties import disk_storage
import train_image_classifier

slim = tf.contrib.slim



flags = tf.app.flags

#State your dataset directory
# flags.DEFINE_string('dataset_dir', disk_storage.LYMPH_DATA_DIR , 'String: Your dataset directory')
#
# flags.DEFINE_string('train_dir', disk_storage.TRAIN_LOGS_DIR , 'String: Your dataset directory')
#
# flags.DEFINE_string('dataset_name', "lymph" , 'String: Your dataset directory')
#
# flags.DEFINE_string('dataset_split_name', 'train' , 'String: Your dataset directory')
#
# flags.DEFINE_string('model_name', 'alexnet' , 'String: Your dataset directory')
#
# #Output filename for the naming the TFRecord file
# flags.DEFINE_string('tfrecord_filename', 'wsi_%s_%05d-of-%05d.tfrecord', 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS



# Selects the 'validation' dataset.
dataset = lymph_biopsy.get_split('validation', disk_storage.LYMPH_DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])

for x in [image, label]:
    print x

