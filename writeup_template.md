## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the line (37) on the file called "find_lane_lines.py". It is used into the verbose_pipeline. This pipeline is used once, to calibrate and generate every pictures I need for the project. 

I start by preparing "object points" and "imgae points", which will be the (x, y, z) coordinates of the chessboard corners. Here I am got 9x6 corners into the chessboard. I used different cheesboard pictures. Each one was took by different angle, then the object points are different for each calibration image.  Thus, `objpoints` will be appended by each cheesboard processed. I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![original_image_korners - copy](https://user-images.githubusercontent.com/19958282/41870026-3606d266-7891-11e8-8990-78f25a660d36.png)
![image_korners - copy](https://user-images.githubusercontent.com/19958282/41870027-365a99fa-7891-11e8-82e1-b198164bf82c.png)

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![original_image_korners - copy](https://user-images.githubusercontent.com/19958282/41870026-3606d266-7891-11e8-8990-78f25a660d36.png)
![image_undistorded - copy](https://user-images.githubusercontent.com/19958282/41870060-4b1b071c-7891-11e8-81b4-e859fb537a21.png)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![image_calc_undistorded](https://user-images.githubusercontent.com/19958282/41870092-6a5b6018-7891-11e8-821b-97c895854062.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (RGB to HLS) and gradient thresholds (Direction and Magnitude) to generate a binary image (thresholding steps at lines # 10 to 54 # in `utilities.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

Magnitude Gradient - Kernel=15, Threshold (30, 100)
![binary_magnitude](https://user-images.githubusercontent.com/19958282/41870335-266fcd0c-7892-11e8-8116-698e2026847a.png)

Direction Gradient - Kernel=15, Threshold (0.7, 1.3)
![binary_direction](https://user-images.githubusercontent.com/19958282/41870329-23375916-7892-11e8-81f8-46129dd4aabc.png)

Sobel X - Threshold (30, 100)
![sobel_x](https://user-images.githubusercontent.com/19958282/41870696-2996a9fa-7893-11e8-8178-c37bc6df6ad1.png)

Sobel Y - Threshold (30, 100)
![sobel_y](https://user-images.githubusercontent.com/19958282/41870700-2b57d0c0-7893-11e8-8355-f7345e6ad057.png)

Combined binary image - SobelX, SobelY, Magnitude Gradient, Directional Gradient
![binary_combined](https://user-images.githubusercontent.com/19958282/41870341-2c85b238-7892-11e8-8339-9738bfe4e329.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
