import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from utilities import calibrate_camera
import glob


dir_name = 'camera_cal\calibration*.jpg'
images_list = glob.glob(dir_name)

#img = cv2.imread(dir_name)

#number of the corners in the x axis and y axis
nx = 9
ny = 6
calibrate_camera(images_list, nx, ny)
# Convert to grayscale
