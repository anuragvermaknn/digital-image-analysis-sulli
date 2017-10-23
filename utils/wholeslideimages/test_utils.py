import cv2, os
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image
import contour_utils as wsi_contour_utils

os.chdir('/Users/pallavgarg/Downloads/Ground_Truth_Extracted/Mask/')

# Resolution level at which slide should be read
resolution_level = 5

# Read a normal slide
normal_wsi_path = 'Normal_001.tif'
# normal_wsi = OpenSlide(normal_wsi_path)
# normal_wsi.level_dimensions
# normal_wsi_sample = normal_wsi.read_region((0, 0), resolution_level, normal_wsi.level_dimensions[resolution_level])

# Read a tumor slide
tumor_wsi_path = 'Tumor_001.tif'
tumor_wsi = OpenSlide(tumor_wsi_path)
tumor_wsi.level_dimensions
tumor_wsi_sample = tumor_wsi.read_region((0, 0), resolution_level, tumor_wsi.level_dimensions[resolution_level])

# Read the corresponding mask file
wsi_mask_path = 'Tumor_001_Mask.tif'
wsi_mask = OpenSlide(wsi_mask_path)

mask_resolution_level = 5
tumor_mask = wsi_mask.read_region((0, 0), mask_resolution_level, wsi_mask.level_dimensions[mask_resolution_level])
# tumor_mask_grayscale = cv2.cvtColor(np.array(tumor_mask), cv2.COLOR_BGR2GRAY)
#
# image, contours, _ = cv2.findContours(tumor_mask_grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# tumor_contour = np.array(tumor_wsi_sample)
# line_color = (0, 255, 0)  # blue color code
# cv2.drawContours(tumor_contour, contours, -1, line_color, 5)

# wsi_contour_utils.get_and_save_patch_samples_from_mask_and_wsi_image(mask_image=tumor_mask,
#                                                                      mask_image_resolution_level=mask_resolution_level,
#                                                                      wsi_path=tumor_wsi_path,
#                                                                      wsi_mask_path=wsi_mask_path,
#                                                                      patch_resolution_level=mask_resolution_level-mask_resolution_level+1)

# below api is working great
wsi_contour_utils.get_and_save_patch_samples_for_non_tumor_images(mask_image_resolution_level=mask_resolution_level,
                                                                  wsi_path=normal_wsi_path)

# below api is working great
wsi_contour_utils.get_and_save_patch_samples_for_tumor_images(mask_image=tumor_mask,
                                                              mask_image_resolution_level=mask_resolution_level,
                                                              wsi_path=tumor_wsi_path,
                                                              wsi_mask_path=wsi_mask_path)
