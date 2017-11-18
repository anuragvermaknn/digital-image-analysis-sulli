#

import getpass

users = ['anurag', 
		 'anuragverma']

user = getpass.getuser()
print('user: %s' % user)

# = {
# 	"anurag": "",
# 	"anuragverma":	""
# }

RAW_DATA_DIR_LIST = {
	"anurag": "",
	"anuragverma":	""
}

CAMELYON_DIR_LIST = {
	"anurag": "/mnt/newhd/camelyon/",
	"anuragverma":	"/Users/anuragverma/Desktop/camelyon/"
}
CAMELYON_DIR = CAMELYON_DIR_LIST[user]

RAW_DATA_DIR = CAMELYON_DIR + "TrainingData/"
RAW_TUMOR_DATA_DIR = RAW_DATA_DIR + "Train_Tumor/"
RAW_NORMAL_DATA_DIR = RAW_DATA_DIR + "Train_Normal/"
RAW_TUMOR_MASK_DIR = CAMELYON_DIR + "Ground_Truth_Extracted/Mask/"

PREPROCESSING_DATA_DIR = CAMELYON_DIR + "PreProcessing/"
PATCHES_DATA_DIR = PREPROCESSING_DATA_DIR + "Patches/"

PATCHES_RAW_DATA_DIR = PATCHES_DATA_DIR + "raw/"

PATCHES_TRAIN_DATA_DIR = PATCHES_DATA_DIR + "train/"
PATCHES_TRAIN_TUMOR_DATA_DIR = PATCHES_TRAIN_DATA_DIR + "tumor/"
PATCHES_TRAIN_NORMAL_DATA_DIR = PATCHES_TRAIN_DATA_DIR + "normal/"

PATCHES_TF_RECORD_DIR = PATCHES_DATA_DIR + "tf_record_directory/"
PATCHES_TRAIN_TF_RECORD_DIR = PATCHES_TF_RECORD_DIR + "train/"


LOGS_DIR = CAMELYON_DIR + "logs/"
TRAIN_LOGS_DIR = LOGS_DIR + "train_logs"


HOME = "/Users/pallavgarg/Documents/freelancing/digital-image-analysis-sulli/utils/wholeslideimages/"
DATA_DIR = HOME + "data/"
LYMPH_DATA_DIR = DATA_DIR + "lymph/"
MNIST_DATA_DIR = DATA_DIR + "mnist/"
TRAIN_DIR = DATA_DIR + "train_directory/"

#LOGS_DIR = DATA_DIR + "logs/"



DIR_FOR_SAVING_NON_TUMOR_PATCHES = TRAIN_DIR + "Normal/"

# Label for Tumor images is 1
DIR_FOR_SAVING_TUMOR_PATCHES = TRAIN_DIR + "Tumor/"

VALIDATION_DIR = DATA_DIR + "validation_directory/"

TF_RECORD_DIR = DATA_DIR + "tf_record_directory/"

labels_file = HOME + "labels.txt"