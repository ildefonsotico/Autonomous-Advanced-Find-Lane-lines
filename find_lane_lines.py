import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from utilities import calibrate_camera, pipeline, convert_RGB_HLS, calculate_undistord, store_RGB_separately, warp, mag_thresh, direction_threshold, abs_sobel_thresh, combining_tecniques
import glob


dir_name = 'camera_cal\calibration*.jpg'
test_img = 'test_images\\test3.jpg'

images_list = glob.glob(dir_name)
test_image = cv2.imread(test_img)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
#img = cv2.imread(dir_name)

#number of the corners in the x axis and y axis
nx = 9
ny = 6
objpoints, imgpoints = calibrate_camera(images_list, nx, ny)
dst = calculate_undistord(test_image, objpoints, imgpoints)
warped = warp(dst)

cv2.imshow('undistord', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
sobel_x = abs_sobel_thresh(dst, 'x', 30, 100)
sobel_y = abs_sobel_thresh(dst, 'y', 30, 100)
mag_bin = mag_thresh(dst,15,(30, 100))
dir_bin = direction_threshold(dst, 15, (0.7, 1.3))
comb = combining_tecniques(sobel_x, sobel_y, mag_bin, dir_bin)
store_RGB_separately(dst)
hls_img = convert_RGB_HLS(dst)
color_merged = pipeline(dst, (90, 200), (30,100))

#img_bin = direction_threshold(img, 5, (30, 100))
# cv2.imshow('Binary', img_bin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Convert to grayscale
