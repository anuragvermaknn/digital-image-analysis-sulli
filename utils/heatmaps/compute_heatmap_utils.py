


import cv2, os
import numpy as np
import tensorflow as tf
from PIL import Image

import properties.wsi_props as wsi_props
from utils.wholeslideimages import contour_utils as wsi_contour_utils
from utils.wholeslideimages import file_utils as wsi_file_utils
import properties.disk_storage as disk_storage_props
import properties.wsi_props as wsi_props
from utils.wholeslideimages import patch_utils, contour_utils


def aggregate_heatmap_probabilities(wsi_name,
                                    heatmap_probabilities_array,
                                    end_point_predictions_batch,
                                    filenames_batch):

    #print (" predictions_batch ", end_point_predictions_batch)
    #print (" predictions_batch shape ", end_point_predictions_batch.shape)
    # Since tumor is class 1 and normal is class 0
    tumor_probabilities = end_point_predictions_batch[:, 1:]
    #print("\n tumor_probabilities ", tumor_probabilities)
    height = heatmap_probabilities_array.shape[1] - 1
    width = heatmap_probabilities_array.shape[0] - 1
    #print ("height ", height, ", width ",width)
    for probability, filename in zip(tumor_probabilities, filenames_batch):
        exact_filename = filename.split('/')[-1]
        #print("exact_filename ",exact_filename)
        # i guess the coordinates are reversed
        x = int(exact_filename.split("_")[0])
        y = int(exact_filename.split("_")[1])
        #print(" x : ", x,", y : ", y)
        #heatmap_probabilities_array[x][y] = float(probability)
        heatmap_probabilities_array[y][x] = float(probability)
        #print("float(probability) ", float(probability))

    return heatmap_probabilities_array

def get_default_heatmap_probabilities_array(wsi_path,
                                            mask_image_resolution_level=None):

    if mask_image_resolution_level is None:
        mask_image_resolution_level = wsi_props.MASK_IMAGE_RESOLUTION_LEVEL

    wsi_image_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_path)

    dimensions = wsi_image_original.level_dimensions[mask_image_resolution_level]
    #print ("dimensions ", dimensions)
    heatmap_probabilities_array = np.zeros((dimensions[1], dimensions[0]), dtype=np.float32)

    return heatmap_probabilities_array

def save_filenames_and_predictions_for_review(wsi_name, filenames_batch, predictions_batch):

    if not (os.path.isdir(disk_storage_props.REVIEW_DIR_FOR_PREDICTIONS)):
        os.makedirs(disk_storage_props.REVIEW_DIR_FOR_PREDICTIONS)

    filepath = disk_storage_props.WSI_REVIEW_FILE_FOR_PREDICTIONS.replace("WSI_NAME", wsi_name)
    review_filename = os.path.join(filepath)
    with tf.gfile.Open(review_filename, 'a') as f:
        for filename, prediction in zip(filenames_batch, predictions_batch):
            exact_filename = filename.split('/')[-1]
            f.write('%s:%d\n' % (exact_filename, prediction))
    f.close()
    return True


def annotate_heatmap_probabilities_array_with_original_mask(heatmap_probabilities_array_image,
                                                            wsi_mask_path,
                                                            mask_image_resolution_level):
    wsi_mask = wsi_file_utils.read_wsi_normal(wsi_normal_path=wsi_mask_path,
                                              resolution_level=mask_image_resolution_level)
    mask_image  = contour_utils.get_grayscale_image_from_rgb(wsi_mask)
    _, contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = np.array(heatmap_probabilities_array_image.copy())
    line_color = (255, 255, 255)
    cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    #  bounding_boxes = contour_utils.get_bbox_from_mask_image(mask_image)
    # for x, y, w, h in bounding_boxes:
    #     cv2.rectangle(rgb_contour, (x, y), (x + w, y + h), (255, 255, 255), 5)

    return rgb_contour

def clean_heatmap_using_morph_ops(heatmap_filepath):

    img_1 = cv2.imread(heatmap_filepath, 0)
    ret, img = cv2.threshold(img_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # dilate the image first
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    # open the images using a larger kernel by 1 pixel
    kernel_4 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
    opening_1 = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel_4)

    kernel_4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    opening_2 = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel_4)
    return opening_1, opening_2