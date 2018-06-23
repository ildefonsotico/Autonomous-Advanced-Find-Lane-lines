import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from utilities import calibrate_camera, measure_curvatures, measure_car_offset, draw_lane, get_line_fit_from_pre_defined, sliding_window, histogram, pipeline, convert_RGB_HLS, calculate_undistord, store_RGB_separately, warp, mag_thresh, direction_threshold, abs_sobel_thresh, combining_tecniques
import glob
from moviepy.editor import VideoFileClip


class Line():
    def __init__(self, n=5):
        self.fit = []
        self.frame = 0
        self.n = n

    def add_fit(self, new_fit):
        if len(self.fit) == self.n:
            del self.fit[0]
        self.fit.append(new_fit)

    def mean_fit(self):
        return np.mean(self.fit, axis=0)

def pipe_verbose():
    dir_name = 'camera_cal\calibration*.jpg'
    test_img = 'test_images\\test3.jpg'

    images_list = glob.glob(dir_name)
    test_image = cv2.imread(test_img)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(dir_name)

    # number of the corners in the x axis and y axis
    nx = 9
    ny = 6
    objpoints, imgpoints = calibrate_camera(images_list, nx, ny, True)
    dst = calculate_undistord(test_image, objpoints, imgpoints, True)
    warp(dst, 'warped.png', True)

    sobel_x = abs_sobel_thresh(dst, 'x', 30, 100, True)
    sobel_y = abs_sobel_thresh(dst, 'y', 30, 100, True)
    mag_bin = mag_thresh(dst, 15, (30, 100), True)
    dir_bin = direction_threshold(dst, 15, (0.7, 1.3), True)
    comb = combining_tecniques(sobel_x, sobel_y, mag_bin, dir_bin, True)
    store_RGB_separately(dst, True)
    hls_img = convert_RGB_HLS(dst, True)
    color_merged, combined_bin = pipeline(dst, (90, 200), (30, 100), True)

    warped, MInv = warp(combined_bin, 'combined_bin_warped.png', True)
    # histogram(warped)

    left_lane_inds, right_lane_inds, left_fit, right_fit, left_fitx, right_fitx = sliding_window(warped, True)
    get_line_fit_from_pre_defined(warped, left_fit, right_fit, 100, True)
    draw_lane(dst, warped, left_fit, right_fit,MInv)

    return objpoints, imgpoints

objpoints, imgpoints = pipe_verbose()

def process_image(img):


    global objpoints
    global imgpoints
    global left_line
    global rigth_line

    dst = calculate_undistord(img, objpoints, imgpoints)
    color_merged, combined_bin = pipeline(dst, (90, 200), (30, 100))
    warped, MInv = warp(combined_bin, 'combined_bin_warped.png')


    if left_line.frame == 0:
        left_lane_inds, right_lane_inds, left_fit, right_fit, left_fitx, right_fitx = sliding_window(warped, True)
    else:
        left_lane_inds, right_lane_inds, left_fit, right_fit, left_fitx, right_fitx = get_line_fit_from_pre_defined(warped, left_line.mean_fit(), rigth_line.mean_fit(), 100, True)

    left_line.add_fit(left_fit)
    rigth_line.add_fit(right_fit)
    left_line.frame += 1
    rigth_line.frame += 1
    img = draw_lane(dst, warped, left_line.mean_fit(), rigth_line.mean_fit(), MInv)
    curvature = np.mean(measure_curvatures(warped, left_line.mean_fit(), rigth_line.mean_fit()))
    car_offset = measure_car_offset(warped, left_line.mean_fit(), rigth_line.mean_fit())
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(img, 'Curvature`s Radius= ' + str(int(curvature)) + '(m)', (50, 100), font, 2, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(img, 'Offset | Vehicle Center| = ' + str(round(car_offset, 2)) + '(m)', (50, 150), font, 2,
                (255, 255, 255), 2, cv2.LINE_AA)
    result = img

    return result



vid_output = 'reg_vid.mp4'

left_line  = Line(7)
rigth_line = Line(7)
# The file referenced in clip1 is the original video before anything has been done to it
clip1 = VideoFileClip("project_video.mp4")

# NOTE: this function expects color images
vid_clip = clip1.fl_image(process_image)
vid_clip.write_videofile(vid_output, audio=False)

