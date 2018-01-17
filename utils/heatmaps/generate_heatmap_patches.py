
#from Queue import Queue
from Queue import *
import time
from threading import Thread


import glob, math
import cv2, os
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image
import utils.wholeslideimages.contour_utils as wsi_contour_utils
from properties import disk_storage as disk_storage_props
from properties import wsi_props
import utils.wholeslideimages.file_utils as wsi_file_utils
import consecutive_patch_utils


tumor_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_TUMOR_DATA_DIR, '*.tif'))
tumor_wsi_paths.sort()
#print(tumor_wsi_paths)
tumor_mask_paths = glob.glob(os.path.join(disk_storage_props.RAW_TUMOR_MASK_DIR, '*.tif'))
tumor_mask_paths.sort()
tumor_image_mask_pairs = list( zip (tumor_wsi_paths, tumor_mask_paths))
#print(tumor_image_mask_pairs)

normal_wsi_paths = glob.glob(os.path.join(disk_storage_props.RAW_NORMAL_DATA_DIR, '*.tif'))
normal_wsi_paths.sort()

mask_image_resolution_level= wsi_props.MASK_IMAGE_RESOLUTION_LEVEL
resolution_level = wsi_props.MASK_IMAGE_RESOLUTION_LEVEL
patch_resolution_level = 0



# def get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps_single_threaded():
#
#     step = 0
#     # for normal_wsi_path in normal_wsi_paths:
#     #     if step >= 1 :
#     #         break
#     #
#     #     success = consecutive_patch_utils.\
#     #         get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps(
#     #         mask_image_resolution_level=mask_image_resolution_level,
#     #         wsi_path=normal_wsi_path,
#     #         patch_resolution_level=patch_resolution_level)
#     #
#     #     print success
#     #     step += 1
#
#     step = 0
#     start_time = time.time()
#     for tumor_wsi_path in tumor_wsi_paths:
#
#         if step >= 10 :
#             break
#
#         success = consecutive_patch_utils.\
#             get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps(
#             mask_image_resolution_level=mask_image_resolution_level,
#             wsi_path=tumor_wsi_path,
#             patch_resolution_level=patch_resolution_level)
#
#         print success
#         step += 1
#     duration = time.time() - start_time
#     print(' raw input patches generated : %d minutes' % math.ceil(duration/60))


class Worker_to_get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps(Thread):
   def __init__(self, queue):
       Thread.__init__(self)
       self.queue = queue

   def run(self):
       while True:
           # Get the work from the queue and expand the tuple
           #directory, link = self.queue.get()
           mask_image_resolution_level, wsi_path, patch_resolution_level = self.queue.get()
           #download_link(directory, link)
           success = consecutive_patch_utils. \
               get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps(
               mask_image_resolution_level=mask_image_resolution_level,
               wsi_path=wsi_path,
               patch_resolution_level=patch_resolution_level)
           self.queue.task_done()

def get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps():
    try:
        ts = time.time()
        # Create a queue to communicate with the worker threads
        queue = Queue()
        # Create 4 worker threads
        for x in range(4):
            worker = Worker_to_get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps(queue)
            # Setting daemon to True will let the main thread exit even though the workers are blocking
            worker.daemon = True
            worker.start()
        step = 0
        start_time = time.time()
        # Put the tasks into the queue as a tuple
        for wsi_path in tumor_wsi_paths:
            step += 1
            if step == 0:
                continue
            wsi_name = wsi_path.split('/')[-1].split('.')[0]
            queue.put((mask_image_resolution_level, wsi_path, patch_resolution_level))
            
            duration = time.time() - start_time
            start_time = time.time()
            print('For ', wsi_name, '\t',' raw input patches generated : %d minutes' % math.ceil(duration / 60))

            if step >= 2:
                break

        # Causes the main thread to wait for the queue to finish processing all the tasks
        queue.join()
        print('Took {}'.format(time.time() - ts))
    except Exception as e:
        print e
        return False
    return True

if __name__ == '__main__':
    get_and_save_consecutive_patch_samples_from_both_images_to_get_heatmaps()
    # print("\n\n\n generate_normal_patches_from_normal_images ")
    # generate_normal_patches_from_normal_images()

    #print("\n\n\n generate_normal_patches_from_tumor_images ")
    #generate_normal_patches_from_tumor_images()
    #
    # print("\n\n\n generate_tumor_patches_from_tumor_images ")
    # generate_tumor_patches_from_tumor_images()
    #print " Patches have been created once."

# class Worker_to_generate_normal_patches_from_tumor_images(Thread):
#    def __init__(self, queue):
#        Thread.__init__(self)
#        self.queue = queue
#
#    def run(self):
#        while True:
#            # Get the work from the queue and expand the tuple
#            #directory, link = self.queue.get()
#            mask_image_resolution_level, tumor_wsi_path, wsi_mask_path = self.queue.get()
#            #download_link(directory, link)
#            wsi_contour_utils.get_and_save_normal_patch_samples_from_both_images(
#                mask_image_resolution_level=mask_image_resolution_level,
#                wsi_path=tumor_wsi_path,
#                wsi_mask_path=wsi_mask_path,
#                is_tumor_image=True)
#            self.queue.task_done()
#
# def generate_normal_patches_from_tumor_images():
#     try:
#         ts = time.time()
#         # Create a queue to communicate with the worker threads
#         queue = Queue()
#         # Create 4 worker threads
#         for x in range(4):
#             worker = Worker_to_generate_normal_patches_from_tumor_images(queue)
#             # Setting daemon to True will let the main thread exit even though the workers are blocking
#             worker.daemon = True
#             worker.start()
#
#         # Important
#         mask_image_resolution_level_special_case = 5
#         # Put the tasks into the queue as a tuple
#         for tumor_wsi_path, wsi_mask_path in tumor_image_mask_pairs:
#             queue.put((mask_image_resolution_level_special_case, tumor_wsi_path, wsi_mask_path))
#             # wsi_contour_utils.get_and_save_normal_patch_samples_from_both_images(mask_image_resolution_level=mask_image_resolution_level,
#             #                                                                      wsi_path=tumor_wsi_path,
#             #                                                                      wsi_mask_path=wsi_mask_path)
#         # Causes the main thread to wait for the queue to finish processing all the tasks
#         queue.join()
#         print('Took {}'.format(time.time() - ts))
#     except Exception as e:
#         print e
#         return False
#     return True
#
#
# class Worker_to_generate_tumor_patches_from_tumor_images(Thread):
#    def __init__(self, queue):
#        Thread.__init__(self)
#        self.queue = queue
#
#    def run(self):
#        while True:
#            # Get the work from the queue and expand the tuple
#            #directory, link = self.queue.get()
#            mask_image_resolution_level, tumor_wsi_path, wsi_mask_path = self.queue.get()
#            #download_link(directory, link)
#            wsi_mask = wsi_file_utils.read_wsi_normal(wsi_normal_path=wsi_mask_path,
#                                                      resolution_level=mask_image_resolution_level)
#            wsi_contour_utils.get_and_save_tumor_patch_samples_for_tumor_images(mask_image=wsi_mask,
#                                                                                mask_image_resolution_level=mask_image_resolution_level,
#                                                                                wsi_path=tumor_wsi_path,
#                                                                                wsi_mask_path=wsi_mask_path)
#            self.queue.task_done()
#
# def generate_tumor_patches_from_tumor_images():
#     try:
#         ts = time.time()
#         # Create a queue to communicate with the worker threads
#         queue = Queue()
#         # Create 4 worker threads
#         for x in range(4):
#             worker = Worker_to_generate_tumor_patches_from_tumor_images(queue)
#             # Setting daemon to True will let the main thread exit even though the workers are blocking
#             worker.daemon = True
#             worker.start()
#         # Put the tasks into the queue as a tuple
#         for tumor_wsi_path, wsi_mask_path in tumor_image_mask_pairs:
#             queue.put((mask_image_resolution_level, tumor_wsi_path, wsi_mask_path))
#             # wsi_mask = wsi_file_utils.read_wsi_normal(wsi_normal_path=wsi_mask_path,resolution_level=mask_image_resolution_level)
#             # wsi_contour_utils.get_and_save_tumor_patch_samples_for_tumor_images(mask_image=wsi_mask,
#             #                                                                     mask_image_resolution_level=mask_image_resolution_level,
#             #                                                                     wsi_path=tumor_wsi_path,
#             #                                                                     wsi_mask_path=wsi_mask_path)
#         # Causes the main thread to wait for the queue to finish processing all the tasks
#         queue.join()
#         print('Took {}'.format(time.time() - ts))
#     except Exception as e:
#         print e
#         return False
#     return True

