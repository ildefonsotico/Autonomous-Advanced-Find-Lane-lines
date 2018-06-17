import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from utilities import calibrate_camera, calculate_undistord, warp
import glob


dir_name = 'camera_cal\calibration*.jpg'
test_img = 'test_images\straight_lines1.jpg'
images_list = glob.glob(dir_name)
img = cv2.imread(test_img)

#img = cv2.imread(dir_name)

#number of the corners in the x axis and y axis
nx = 9
ny = 6
objpoints, imgpoints = calibrate_camera(images_list, nx, ny)
dst = calculate_undistord(img, objpoints, imgpoints, True)
warped = warp(dst)
# Convert to grayscale
