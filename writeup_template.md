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
![original_cheesboard](https://user-images.githubusercontent.com/19958282/41874689-fa40e79a-789e-11e8-8638-46ca3a54d8d8.png)
![korners_cheesboard](https://user-images.githubusercontent.com/19958282/41874674-f1544b68-789e-11e8-9534-173ccff81a43.png)

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![cheesboard_undistorted](https://user-images.githubusercontent.com/19958282/41874733-15666572-789f-11e8-9707-702feca7810c.png)


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![undistorted_image](https://user-images.githubusercontent.com/19958282/41874822-5f505c92-789f-11e8-8fd1-b2255091e1fb.png)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (RGB to HLS) and gradient thresholds (Direction and Magnitude) to generate a binary image (thresholding steps at lines # 10 to 54 # in `utilities.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

##### Magnitude Gradient - Kernel=15, Threshold (30, 100)
![magnitude_gradient](https://user-images.githubusercontent.com/19958282/41874900-a34e5eda-789f-11e8-8136-fa795f034bba.png)

##### Direction Gradient - Kernel=15, Threshold (0.7, 1.3)
![directional_gradient](https://user-images.githubusercontent.com/19958282/41875004-ef764930-789f-11e8-9045-ab5902d527d8.png)

##### Sobel X - Threshold (30, 100)
![sobelx](https://user-images.githubusercontent.com/19958282/41875146-5e994646-78a0-11e8-83c1-76594f41ee3f.png)

##### Sobel Y - Threshold (30, 100)
![sobely](https://user-images.githubusercontent.com/19958282/41875206-8e7b5a34-78a0-11e8-9e7a-9a7848a1dac4.png)

##### Combined binary image - SobelX, SobelY, Magnitude Gradient, Directional Gradient
![combined_binary](https://user-images.githubusercontent.com/19958282/41875266-b992d29c-78a0-11e8-91ee-985a89a21d08.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 118 through 143 in the file `utilities.py` (/utilities.py).  The `warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
#source image
 src = np.float32([[690, 450],
                   [1040, 685],
                   [250, 690],
                   [590, 450]]
                   )
#destination image
dst = np.float32([[980, 0], [980, 720], [320, 720], [320, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 690, 450      | 980, 0        | 
| 1040, 685      | 980, 720      |
| 250, 690     | 320, 720      |
| 590, 450      | 320, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![combined_bin_warped](https://user-images.githubusercontent.com/19958282/41871245-bf18bc06-7894-11e8-942b-eb100ce0eafc.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
![sliding_pre_defined_window](https://user-images.githubusercontent.com/19958282/41871418-48720156-7895-11e8-844c-6d7f6114ea55.png)
![sliding_window](https://user-images.githubusercontent.com/19958282/41871420-48be732e-7895-11e8-9307-93682b7e386f.png)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # 84 through 86# in my code in `find_lane_lines.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)
