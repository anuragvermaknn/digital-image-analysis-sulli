
import glob
import cv2, os
import numpy as np
import tensorflow as tf
from PIL import Image

import properties.wsi_props as wsi_props
from utils.wholeslideimages import contour_utils as wsi_contour_utils
from utils.wholeslideimages import file_utils as wsi_file_utils
import properties.disk_storage as disk_storage_props
import properties.wsi_props as wsi_props
from utils.wholeslideimages import patch_utils

def get_and_save_consecutive_patches_from_both_images(wsi,
                                                     wsi_name,
                                                     wsi_image_original,
                                                     mask_image_resolution_level,
                                                     patch_resolution_level):

    # get mask which should contain most of the red region of wsi
    saturation_thresholded_mask = wsi_contour_utils.get_saturation_thresholded_mask_from_non_tumor_wsi(non_tumor_wsi=wsi)
    saturation_thresholded_mask_grayscale = wsi_contour_utils.get_saturation_thresholded_mask_from_non_tumor_wsi_grayscale(non_tumor_wsi=wsi)

    stride_for_consecutive_samples = wsi_props.STRIDE_FOR_CONSECUTIVE_SAMPLES_FOR_EVALUATION
    # get bboxes which cover the mask contours, and
    # consecutive samples of patch starting points
    consecutive_samples_of_patch_starting_points, rgb_image_with_bbox = wsi_contour_utils. \
        get_consecutive_samples_of_patch_starting_points_and_image_with_bbox_with_stride(mask_image=saturation_thresholded_mask,
                                                                                         stride=stride_for_consecutive_samples,
                                                                                         rgb_image=np.array(wsi.copy()))

    if rgb_image_with_bbox is not None:
        #Image.fromarray(np.array(rgb_image_with_bbox)).show()
        if not (os.path.isdir(disk_storage_props.HEATMAP_REVIEW_DIR)):
            os.makedirs(disk_storage_props.HEATMAP_REVIEW_DIR)
        if not (os.path.isdir(disk_storage_props.REVIEW_DIR_FOR_BBOX_SELECTED_TO_GET_INPUT_FOR_HEATMAP)):
            os.makedirs(disk_storage_props.REVIEW_DIR_FOR_BBOX_SELECTED_TO_GET_INPUT_FOR_HEATMAP)
        # For Review : bbox actually selected
        bbox_review_filepath = disk_storage_props.WSI_REVIEW_FILE_FOR_BBOX_SELECTED_TO_GET_INPUT_FOR_HEATMAP.replace("WSI_NAME", wsi_name)
        if not os.path.exists(bbox_review_filepath):
            Image.fromarray(np.array(rgb_image_with_bbox)).save(fp=bbox_review_filepath, format="PNG")


    dir_for_consecutive_patches = disk_storage_props.RAW_PATCHES_DIR_TO_GET_HEATMAPS
    if not (os.path.isdir(dir_for_consecutive_patches)):
        os.makedirs(dir_for_consecutive_patches)

    # earlier
    #dir_for_consecutive_patches_for_given_wsi = dir_for_consecutive_patches + str(wsi_name) + "/"
    # now
    dir_for_consecutive_patches_for_given_wsi = disk_storage_props.WSI_RAW_PATCHES_DIR_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
    if not (os.path.isdir(dir_for_consecutive_patches_for_given_wsi)):
        os.makedirs(dir_for_consecutive_patches_for_given_wsi)

    if patch_resolution_level is None:
        patch_resolution_level = 0
    #print "patch_resolution_level : {}".format(patch_resolution_level)
    scaling_factor = pow(2, mask_image_resolution_level - patch_resolution_level)

    print "Number of consecutive samples : {}".format(len(consecutive_samples_of_patch_starting_points))
    #Image.fromarray(np.array(wsi)).show()

    samples_rejected = 0
    samples_accepted = 0
    rgb_contour = np.array(wsi.copy())
    for x, y in consecutive_samples_of_patch_starting_points:

        # if samples_accepted > 55:
        #     continue

        if(saturation_thresholded_mask_grayscale[y,x] != 0) :
        # red_pixels_ratio_threshold = 0.5
        # if(patch_utils.criteria_for_normal_patch_selection_in_non_tumor_images_with_threshold(patch_read_from_wsi_at_zero_level=patch_to_be_saved,
        #                                                                        red_pixels_ratio_threshold=red_pixels_ratio_threshold)):

            #print("rgb_contour[y][x]\n",rgb_contour[y][x])
            patch_to_be_saved = wsi_image_original. \
                read_region((x * scaling_factor, y * scaling_factor),
                            patch_resolution_level,
                            (wsi_props.PATCH_SAMPLE_BOX_SIZE, wsi_props.PATCH_SAMPLE_BOX_SIZE))

            filepath_for_current_patch_for_given_wsi = \
                dir_for_consecutive_patches_for_given_wsi + str(x) + "_" + str(y) + "_" + str(patch_resolution_level) + ".PNG"


            patch_to_be_saved.save(fp=filepath_for_current_patch_for_given_wsi)
            patch_to_be_saved.close()
            #print " saturation_thresholded_mask_grayscale value : {}".format(saturation_thresholded_mask_grayscale[y,x])
            cv2.rectangle(rgb_contour, (x, y), (x + 1, y + 1), (255, 255, 255), 1)
            samples_accepted += 1
        else :
            samples_rejected += 1

    print " samples_accepted : {0}, \t\t samples_rejected : {1}".format(samples_accepted, samples_rejected)

    dir_for_heatmap_patches_review = disk_storage_props.HEATMAP_REVIEW_DIR
    if not (os.path.isdir(dir_for_heatmap_patches_review)):
        os.makedirs(dir_for_heatmap_patches_review)

    # For Review : bbox starting points actually accepted
    bbox_accepted_review_filepath = disk_storage_props.WSI_REVIEW_FILE_FOR_BBOX_ACCEPTED_TO_GET_INPUT_FOR_HEATMAP.replace(
        "WSI_NAME", wsi_name)
    if not os.path.exists(bbox_accepted_review_filepath):
        Image.fromarray(np.array(rgb_contour)).save(fp=bbox_accepted_review_filepath, format="PNG")

    patches_count_filepath = disk_storage_props.WSI_RAW_PATCHES_COUNT_FILE_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
    _write_patches_count_file(filepath=patches_count_filepath,
                             patches_count=samples_accepted)
    print ("read_patches_count_file ",read_patches_count_file(wsi_name))

    return True

def get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps(mask_image_resolution_level, wsi_path, wsi_mask_path=None,
                                                       patch_resolution_level=None, is_tumor_image=False):

    wsi_name = wsi_path.split('/')[-1].split('.')[0]
    if wsi_mask_path is not None:
        wsi_mask_path_name = wsi_mask_path.split('/')[-1].split('.')[0]
        print "\n Processing Image id : ", wsi_name, " And Image Mask id : ", wsi_mask_path_name
    else:
        print "\n Processing Image id : ", wsi_name

    wsi_image_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_path)

    # read the wsi at the mask resolution level
    # in order to compute mask from it,
    # which should contain most of the red region of wsi
    wsi_sampled = wsi_image_original.\
        read_region((0, 0), mask_image_resolution_level,
                    wsi_image_original.level_dimensions[mask_image_resolution_level])


    try:
        get_and_save_consecutive_patches_from_both_images(wsi=wsi_sampled,
                                                          wsi_name=wsi_name,
                                                          wsi_image_original=wsi_image_original,
                                                          mask_image_resolution_level=mask_image_resolution_level,
                                                          patch_resolution_level=patch_resolution_level)

    except Exception as e:
        print e
        return False

    return True


def _write_patches_count_file(filepath, patches_count):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  counts_filename = os.path.join(filepath)
  with tf.gfile.Open(counts_filename, 'w') as f:
      f.write('%d' % patches_count)


def read_patches_count_file(wsi_name):
    patches_count = None
    patches_count_filename = disk_storage_props.WSI_RAW_PATCHES_COUNT_FILE_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
    print ("patches_count_filename ", patches_count_filename)
    with tf.gfile.Open(patches_count_filename, 'r') as f:
        patches_count = f.readline()
        f.close()

    return int(patches_count)

def plot_bbox_accepted(wsi_path, mask_image_resolution_level = None):

    wsi_name = wsi_path.split('/')[-1].split('.')[0]

    if mask_image_resolution_level is None:
        mask_image_resolution_level = wsi_props.MASK_IMAGE_RESOLUTION_LEVEL

    wsi_image_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_path)
    wsi = wsi_image_original.\
        read_region((0, 0), mask_image_resolution_level,
                    wsi_image_original.level_dimensions[mask_image_resolution_level])

    bbox_accepted_dir_for_wsi = disk_storage_props.WSI_RAW_PATCHES_DIR_TO_GET_HEATMAPS.replace("WSI_NAME", wsi_name)
    filenames = glob.glob(os.path.join(bbox_accepted_dir_for_wsi, '*.PNG'))
    filenames.sort()

    rgb_contour = np.array(wsi.copy())
    #print("\nfilenames ", filenames)
    for filename in filenames:
        exact_filename = filename.split('/')[-1]
        #print("exact_filename ",exact_filename)
        # i guess the coordinates are reversed
        x = int(exact_filename.split("_")[0])
        y = int(exact_filename.split("_")[1])
        cv2.rectangle(rgb_contour, (x, y), (x + 5, y + 5), (0, 255, 0), 5)

    bbox_accepted_review_filepath = disk_storage_props.WSI_REVIEW_FILE_FOR_BBOX_ACCEPTED_TO_GET_INPUT_FOR_HEATMAP.replace(
        "WSI_NAME", wsi_name)
    if not os.path.exists(bbox_accepted_review_filepath):
        Image.fromarray(np.array(rgb_contour)).save(fp=bbox_accepted_review_filepath, format="PNG")


if __name__ == '__main__':
    wsi_name = "Tumor_001"
    tumor_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_TUMOR_DATA_DIR, '*.tif'))
    tumor_wsi_paths.sort()
    wsi_path = tumor_wsi_paths[0]
    plot_bbox_accepted(wsi_path=wsi_path)
