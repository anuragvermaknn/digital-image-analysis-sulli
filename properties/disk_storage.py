#

HOME = "/Users/pallavgarg/Documents/freelancing/digital-image-analysis-sulli/utils/wholeslideimages/"
DATA_DIR = HOME + "data/"
LYMPH_DATA_DIR = DATA_DIR + "lymph/"
MNIST_DATA_DIR = DATA_DIR + "mnist/"
TRAIN_DIR = DATA_DIR + "train_directory/"

LOGS_DIR = DATA_DIR + "logs/"
TRAIN_LOGS_DIR = LOGS_DIR + "train_logs"


DIR_FOR_SAVING_NON_TUMOR_PATCHES = TRAIN_DIR + "Normal/"

# Label for Tumor images is 1
DIR_FOR_SAVING_TUMOR_PATCHES = TRAIN_DIR + "Tumor/"

VALIDATION_DIR = DATA_DIR + "validation_directory/"

TF_RECORD_DIR = DATA_DIR + "tf_record_directory/"

labels_file = HOME + "labels.txt"