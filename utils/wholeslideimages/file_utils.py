# This file contains the following APIs :
#
#

import cv2, os
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image

# get openslide object from the wsi path
def get_wsi_openslide_object(wsi_path):

    try:
        wsi_openslide_object = OpenSlide(wsi_path)
        return wsi_openslide_object
    except Exception as e:
        return None

# read wsi at given resolution level from the path
def read_wsi_normal(wsi_normal_path,
                    resolution_level=None):

    wsi_normal = OpenSlide(wsi_normal_path)
    if resolution_level is None:
        resolution_level = wsi_normal.level_count - 1
    wsi_normal_sample = wsi_normal.read_region((0, 0), resolution_level, wsi_normal.level_dimensions[resolution_level])
    wsi_normal.close()
    return wsi_normal_sample

# read both tumor wsi and it's mask from their paths
def read_wsi_tumor(wsi_tumor_path,
                   wsi_mask_path,
                   resolution_level=None):

    try :

        wsi_tumor = OpenSlide(wsi_tumor_path)
        if resolution_level is None:
            resolution_level = wsi_tumor.level_count - 1
        wsi_tumor_sample = wsi_tumor.read_region((0, 0), resolution_level, wsi_tumor.level_dimensions[resolution_level])
        wsi_tumor.close()

        wsi_tumor_mask = OpenSlide(wsi_mask_path)
        wsi_tumor_mask_sample = wsi_tumor_mask.read_region((0, 0), resolution_level, wsi_tumor_mask.level_dimensions[resolution_level])
        wsi_tumor_mask.close()
    except Exception as e:
        return None, None

    return wsi_tumor_sample, wsi_tumor_mask_sample

