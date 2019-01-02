"""
This project is used to enhance the contrast of X-ray image. It is an implementation of Morphological Transformation.
For detail, please refer to 'Medical Image Enhancement Using Morphological Transformation'.

Last Modified: Jan 1st, 2019

vxallset@outlook.com

All rights reserved.
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import support
import utils

print('Start processing...')

start_time = time.time()

file_name = '3.raw'
file_path = './demo_img/' + file_name
img_save_path = './demo_img/'


raw_image = utils.raw_read(file_path)

roi_image = support.get_ROI(raw_image)

kernel_size = 34

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
ABH_img = cv2.morphologyEx(roi_image, cv2.MORPH_BLACKHAT, kernel)
ATH_img = cv2.morphologyEx(roi_image, cv2.MORPH_TOPHAT, kernel)

A_q = roi_image + ATH_img - ABH_img

result = support.post_propossing(A_q, 2.2)
end_time = time.time()
print('Finished processing, time elapsed: {}s.'.format(end_time - start_time))

original_image_complement_with_gamma = support.post_propossing(roi_image, 2.2)
plt.imsave(img_save_path + file_name[:-4] + '_original.png', original_image_complement_with_gamma, cmap='gray')
plt.imsave(img_save_path + file_name[:-4] + '_enhanced.png', result, cmap='gray')

utils.visualize_center_line_density((original_image_complement_with_gamma, result),
                                    titles=['Original Image', 'Morphological Transformation Enhanced Image'])
plt.show()