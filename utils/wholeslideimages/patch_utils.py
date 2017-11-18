# This file contains the following APIs :
#
# criteria_for_patch_selection_in_tumor_images
# criteria_for_patch_selection_in_non_tumor_images
# :
#

import cv2, os
import numpy as np
from PIL import Image
import file_utils as wsi_file_utils
import properties.wsi_props as wsi_props
import contour_utils

# mask_image,                     # rgb image of mask
# mask_image_resolution_level,     # resolution level of mask image
# wsi_path,                        # path of the original wsi
# wsi_mask_path,                   # path of the original wsi mask
# patch_resolution_level,          # resolution level for patch extraction
# patch):                          # patch


# In a good sample, the patch square box should
# cover tumor region mostly. Hence, only less number of
# pixels should lie outside the contour of tumors in mask path
def criteria_for_tumor_patch_selection_in_tumor_images(patch_read_from_mask_at_zero_level):   # corresponding patch from mask

    grayscale_image = contour_utils.get_grayscale_image_from_rgb(patch_read_from_mask_at_zero_level)
    count_tumor_pxls_in_patch_section = cv2.countNonZero(np.array(grayscale_image)) * 1.0

    total_pxls_in_square_box = wsi_props.PATCH_SAMPLE_BOX_SIZE * wsi_props.PATCH_SAMPLE_BOX_SIZE
    ratio_tumor_pxls_in_patch_section  =  count_tumor_pxls_in_patch_section/ total_pxls_in_square_box
    #print "ratio_tumor_pxls_in_patch_section : {0:.2%}".format(ratio_tumor_pxls_in_patch_section)

    if(ratio_tumor_pxls_in_patch_section
        > wsi_props.PATCH_TUMOR_PIXELS_RATIO_THRESHOLD):
        #print "ratio_tumor_pxls_in_patch_section : {0:.2%}".format(ratio_tumor_pxls_in_patch_section)
        return True
    else:
        False

# In a good sample, the patch square box should contain
# a lot of red pixels. Hence, only less number of pixels
# should have non-red hue.
def criteria_for_normal_patch_selection_in_non_tumor_images(patch_read_from_wsi_at_zero_level):    # patch extracted from wsi original

    hsv = cv2.cvtColor(np.array(patch_read_from_wsi_at_zero_level), cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    red_mask_on_hsv = cv2.inRange(hsv, lower_red, upper_red)

    count_red_pixels_in_patch = cv2.countNonZero(np.array(red_mask_on_hsv)) * 1.0
    total_pxls_in_square_box = wsi_props.PATCH_SAMPLE_BOX_SIZE * wsi_props.PATCH_SAMPLE_BOX_SIZE
    ratio_red_pxls_in_patch_section = count_red_pixels_in_patch/ total_pxls_in_square_box
    #print "ratio_red_pxls_in_patch_section : ",ratio_red_pxls_in_patch_section
    if( ratio_red_pxls_in_patch_section
        > wsi_props.PATCH_NON_TUMOR_RED_PIXELS_RATIO_THRESHOLD):
        #print "ratio_red_pxls_in_patch_section : {0:.2%}".format(ratio_red_pxls_in_patch_section)
        return True
    else:
        return False





