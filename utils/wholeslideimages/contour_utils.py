# This file contains the following APIs :
#
#           BOUNDING BOX APIs
#
# get_grayscale_image_from_rgb(rgb_image):
# get_external_contours_from_grayscale_image(grayscale_image):
# get_bbox_from_contours(contours):
# get_bbox_from_mask_image(mask_image):
# get_random_samples_of_patch_starting_points(mask_image):
#
#           PATCH APIs
#
# get_and_save_patch_samples_from_mask_and_wsi_image:
#
# :

import cv2, os
import numpy as np
from PIL import Image
import file_utils as wsi_file_utils
import properties.wsi_props as wsi_props
import properties.disk_storage as disk_storage_props
import patch_utils


# To find external contours, it is necessary to convert
# the RGB image to a gray scale image
def get_grayscale_image_from_rgb(rgb_image):

    image = rgb_image
    if type(image) is not np.ndarray:
        image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# get external contours from a gray scale image
# simple chain approximation retrieved
def get_external_contours_from_grayscale_image(grayscale_image):

    image, contours, hierarchy = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# get a list of bounding boxes which encloses
# the given contours
def get_bbox_from_contours(contours):

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes

# get a list of bounding boxes which encloses
# all the contours in a RGB image
def get_bbox_from_mask_image(mask_image):

    grayscale_image = get_grayscale_image_from_rgb(mask_image)
    contours = get_external_contours_from_grayscale_image(grayscale_image)
    bounding_boxes = get_bbox_from_contours(contours)
    return bounding_boxes

# get random points within the bounding boxes (which encloses
# the contours in the image) from the mask image
# Note :
# 1. In case of tumor images, the mask_image is already provided
# 2. In case of non tumor images, the mask_image has to computed
# from non_tumor_wsi upon thresholding it's saturation counterpart image
def get_random_samples_of_patch_starting_points(mask_image):

    bounding_boxes = get_bbox_from_mask_image(mask_image)
    list_starting_points = []
    for x, y, w, h in bounding_boxes:
        #print " x, y, w, h : {0}, {1}, {2}, {3}".format(x, y, w, h)
        X = np.random.random_integers(x, x+w, wsi_props.PATCH_SAMPLING_SIZE_INSIDE_BBOX)
        Y = np.random.random_integers(y, y+h, wsi_props.PATCH_SAMPLING_SIZE_INSIDE_BBOX)
        # append in the list
        list_starting_points += zip(X, Y)
    return list_starting_points

def get_saturation_thresholded_mask_from_non_tumor_wsi(non_tumor_wsi):

    rgb_image = np.array(non_tumor_wsi)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    s2 = cv2.medianBlur(s, 25)
    ret, saturation_thresholded_mask = cv2.threshold(s2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print ret
    #ret2, saturation_thresholded_mask2 = cv2.threshold(saturation_thresholded_mask, 0, 255, cv2.THRESH_BINARY)
    #print ret2, type(saturation_thresholded_mask2)
    saturation_thresholded_mask_rgb = cv2.cvtColor(saturation_thresholded_mask, cv2.COLOR_GRAY2BGR)
    return saturation_thresholded_mask_rgb


# Given a 1. mask image, 2. it's resolution level, 3. path of the wsi image,
# a). extract random patches within the bounding boxes (which encloses
# the contours in the mask images), from the original wsi path at
# the given 4. resolution level.
# b). Save the patches at the appropriate path
# def get_and_save_patch_samples_from_mask_and_wsi_image(mask_image,          # rgb image of mask
#                                               mask_image_resolution_level,  # resolution level of mask image
#                                               wsi_path,                     # path of the original wsi image
#                                               wsi_mask_path,                # path of the original wsi mask
#                                               patch_resolution_level=None): # resolution level for patch extraction
#
#     patch_starting_points = get_random_samples_of_patch_starting_points(mask_image)
#     if patch_resolution_level is None:
#         patch_resolution_level = 0
#     print "patch_resolution_level : {}".format(patch_resolution_level)
#     scaling_factor = pow(2, mask_image_resolution_level - patch_resolution_level)
#     print " scaling_factor : {} ".format(scaling_factor)
#     wsi_image_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_path)
#     wsi_mask_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_mask_path)
#
#     count = 0
#     for x,y in patch_starting_points:
#         patch = wsi_image_original.read_region((x*scaling_factor, y*scaling_factor),
#                                                patch_resolution_level,
#                                                (wsi_props.PATCH_SAMPLE_BOX_SIZE,wsi_props.PATCH_SAMPLE_BOX_SIZE))
#         patch_from_mask = wsi_mask_original.read_region((x*scaling_factor, y*scaling_factor),
#                                                patch_resolution_level,
#                                                (wsi_props.PATCH_SAMPLE_BOX_SIZE,wsi_props.PATCH_SAMPLE_BOX_SIZE))
#
#         patch.save(fp="/Users/pallavgarg/Documents/freelancing/digital-image-analysis-sulli/utils/wholeslideimages/patches_test_samples/"+str(count)+"_1.png",
#                    format='PNG')
#         patch.close()
#         count += 1
#     wsi_image_original.close()
#     return True


# Given a 1. mask image, 2. it's resolution level, 3. path of the wsi image,
# a). extract random patches within the bounding boxes (which encloses
# the contours in the mask images), from the original wsi path at
# the given 4. resolution level.
# b). Save the patches at the appropriate path
def get_and_save_patch_samples_from_mask_and_wsi_image(mask_image,          # rgb image of mask
                                              mask_image_resolution_level,  # resolution level of mask image
                                              wsi_path,                     # path of the original wsi image
                                              wsi_mask_path=None,           # path of the original wsi mask
                                              patch_resolution_level=None, # resolution level for patch extraction
                                              is_tumor_image=True):

    patch_starting_points = get_random_samples_of_patch_starting_points(mask_image)
    print "patch_starting_points computed"
    if patch_resolution_level is None:
        patch_resolution_level = 0
    #print "patch_resolution_level : {}".format(patch_resolution_level)
    scaling_factor = pow(2, mask_image_resolution_level - patch_resolution_level)
    #print " scaling_factor : {} ".format(scaling_factor)

    wsi_image_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_path)
    wsi_mask_original = None

    if is_tumor_image:
        wsi_mask_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_mask_path)

    count = 0
    samples_accepted = 0
    samples_rejected = 0
    for x,y in patch_starting_points:

        save_patch = False
        patch_to_be_saved = None
        # For Tumor Images
        if is_tumor_image :
            patch_read_from_mask_at_zero_level = wsi_mask_original.read_region((x*scaling_factor, y*scaling_factor),
                                                   0,
                                                   (wsi_props.PATCH_SAMPLE_BOX_SIZE,wsi_props.PATCH_SAMPLE_BOX_SIZE))

            # Is this patch a good sample ?
            if patch_utils.criteria_for_patch_selection_in_tumor_images(patch_read_from_mask_at_zero_level):
                save_patch = True
                patch_to_be_saved = wsi_image_original.read_region(
                    (x * scaling_factor, y * scaling_factor),
                    0,
                    (wsi_props.PATCH_SAMPLE_BOX_SIZE, wsi_props.PATCH_SAMPLE_BOX_SIZE))

                dir_for_saving_tumor_patches = disk_storage_props.DIR_FOR_SAVING_TUMOR_PATCHES
                if not ( os.path.isdir(dir_for_saving_tumor_patches)):
                    os.makedirs(dir_for_saving_tumor_patches)

                patch_to_be_saved.save(fp=dir_for_saving_tumor_patches+str(count)+".PNG")
                patch_to_be_saved.close()
                samples_accepted += 1
            else:
                samples_rejected += 1

            patch_read_from_mask_at_zero_level.close()

        else:
            patch_read_from_wsi_at_zero_level = wsi_image_original.read_region((x*scaling_factor, y*scaling_factor),
                                               0,
                                               (wsi_props.PATCH_SAMPLE_BOX_SIZE,wsi_props.PATCH_SAMPLE_BOX_SIZE))

            # Is this patch a good sample ?
            if patch_utils.criteria_for_patch_selection_in_non_tumor_images(patch_read_from_wsi_at_zero_level):
                print " Criteria satisfied by patch"
                save_patch=True
                patch_to_be_saved = patch_read_from_wsi_at_zero_level
                dir_for_saving_non_tumor_patches = disk_storage_props.DIR_FOR_SAVING_NON_TUMOR_PATCHES
                if not (os.path.isdir(dir_for_saving_non_tumor_patches)):
                    os.makedirs(dir_for_saving_non_tumor_patches)

                patch_to_be_saved.save(fp=dir_for_saving_non_tumor_patches+str(count)+".PNG")
                patch_to_be_saved.close()
                samples_accepted += 1
            else:
                samples_rejected += 1
                #print " Criteria UN-satisfied by patch"


        count += 1
    wsi_image_original.close()

    print " samples_accepted : {0}, \t\t samples_rejected : {1}".format(samples_accepted, samples_rejected)
    if is_tumor_image:
        wsi_mask_original.close()
    return True



def get_and_save_patch_samples_for_tumor_images(mask_image,                 # rgb image of mask
                                              mask_image_resolution_level,  # resolution level of mask image
                                              wsi_path,                     # path of the original wsi image
                                              wsi_mask_path,           # path of the original wsi mask
                                              patch_resolution_level=None): # resolution level for patch extraction

    try :
        get_and_save_patch_samples_from_mask_and_wsi_image(mask_image=mask_image,
                                                       mask_image_resolution_level=mask_image_resolution_level,
                                                       wsi_path=wsi_path,
                                                       wsi_mask_path=wsi_mask_path,
                                                       patch_resolution_level=patch_resolution_level,
                                                       is_tumor_image=True)
    except Exception as e:
        return False

    return True

# In case of non tumor images, the mask_image has to computed
# from non_tumor_wsi upon thresholding it's saturation counterpart image
def get_and_save_patch_samples_for_non_tumor_images(mask_image_resolution_level,  # resolution level of mask image
                                              wsi_path,                     # path of the original wsi image
                                              patch_resolution_level=None): # resolution level for patch extraction

    wsi_image_original = wsi_file_utils.get_wsi_openslide_object(wsi_path=wsi_path)

    non_tumor_wsi = wsi_image_original.read_region((0, 0), mask_image_resolution_level, wsi_image_original.level_dimensions[mask_image_resolution_level])

    mask_image = get_saturation_thresholded_mask_from_non_tumor_wsi(non_tumor_wsi=non_tumor_wsi)

    try :
        # No path for wsi mask this time
        get_and_save_patch_samples_from_mask_and_wsi_image(mask_image=mask_image,
                                                       mask_image_resolution_level=mask_image_resolution_level,
                                                       wsi_path=wsi_path,
                                                       wsi_mask_path=None,
                                                       patch_resolution_level=patch_resolution_level,
                                                       is_tumor_image=False)
    except Exception as e:
        print e
        return False

    return True

