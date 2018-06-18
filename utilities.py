import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

out_path='output_images\\'

def direction_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    plt.imshow(binary_output, cmap='gray')
    plt.savefig(out_path + 'binary_direction.png')
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    plt.imshow(sobelx, cmap='gray')
    plt.savefig(out_path+'sobel_x.png')
    plt.imshow(sobely, cmap='gray')
    plt.savefig(out_path+'sobel_y.png')

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    plt.imshow(binary_output, cmap='gray')
    plt.savefig(out_path+'binary_magnitude.png')
    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
def calibrate_camera(images, nx=9, ny=6, verbose=False):

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for file_name in enumerate(images):

        img = cv2.imread(file_name[1])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        #print (ret)
        #print (corners)
        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # cv2.imshow('chessboardcorner', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    for file_name in enumerate(images):
        img = cv2.imread(file_name[1])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        plt.imshow(dst)
        plt.savefig(out_path + 'image_undistorded.png')
        if(verbose):
            cv2.imshow('undistord', dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return objpoints, imgpoints

def calculate_undistord(img, objpoints, imgpoints, verbose=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.savefig(out_path + 'image_calc_undistorded.png')
    if (verbose):
        cv2.imshow('undistord', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return dst
def warp(img, verbose=False):
    img_size = (img.shape[1], img.shape[0])

    #source image
    src = np.float32([[690, 450],
                     [1040, 685],
                     [250, 690],
                     [590, 450]]
                     )

    #destination image
    dst = np.float32([[980, 0], [980, 720], [320, 720], [320, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    if (verbose):
        cv2.imshow('warped', warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return warped

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        cv2.imwrite('sobelx.png', abs_sobel)
        plt.imshow(abs_sobel, cmap='gray')
        plt.savefig('abs_sobel_x.png')
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        plt.imshow(abs_sobel, cmap='gray')
        plt.savefig('abs_sobel_y.png')


    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    plt.imshow(binary_output, cmap='gray')
    plt.savefig('binary_matplot.png')
    # Return the result
    return binary_output

def combining_tecniques(gradx,grady, mag_binary, dir_binary):
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    plt.imshow(combined, cmap='gray')
    plt.savefig(out_path + 'binary_combined.png')
    return combined

def store_RGB_separately(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    plt.imshow(R)
    plt.savefig(out_path + 'red_channel.png')
    plt.imshow(G)
    plt.savefig(out_path + 'green_channel.png')
    plt.imshow(B)
    plt.savefig(out_path + 'blue_channel.png')

def convert_RGB_HLS(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    binary_S(S)
    binary_H(H)
    binary_L(L)

    plt.imshow(H)
    plt.savefig(out_path + 'h_channel.png')
    plt.imshow(L)
    plt.savefig(out_path + 'l_channel.png')
    plt.imshow(S)
    plt.savefig(out_path + 's_channel.png')
    return hls

def binary_S(s):
    thresh = (90, 255)
    binary = np.zeros_like(s)
    binary[(s > thresh[0]) & (s <= thresh[1])] = 1
    plt.imshow(binary, cmap='gray')
    plt.savefig(out_path + 's_binary_threshold_channel.png')

def binary_H(h):
    thresh = (15, 100)
    binary = np.zeros_like(h)
    binary[(h > thresh[0]) & (h <= thresh[1])] = 1
    plt.imshow(binary, cmap='gray')
    plt.savefig(out_path + 'h_binary_threshold_channel.png')

def binary_L(l):
    thresh = (30, 100)
    binary = np.zeros_like(l)
    binary[(l > thresh[0]) & (l <= thresh[1])] = 1
    plt.imshow(binary, cmap='gray')
    plt.savefig(out_path + 'l_binary_threshold_channel.png')


def pipeline(img, s_thresh=(120, 255), sx_thresh=(50, 150)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    plt.imshow(color_binary)
    plt.savefig(out_path + 'pipeline_stack_gradient_color.png')
    plt.imshow(combined_binary, cmap='gray')
    plt.savefig(out_path + 'pipeline_combined_gradient_color.png')
    return color_binary, combined_binary