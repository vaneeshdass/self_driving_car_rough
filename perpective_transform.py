# from color_trans_gradients import *

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# Loading camera calibration
calibration_parameters = pickle.load(open('serialized_camera_data/camera_calibration.p', 'rb'))
mtx, dist = map(calibration_parameters.get, ('mtx', 'dist'))

# Load calibration images.
test_images_straight_lines = list(map(lambda imageFileName: (imageFileName, cv2.imread(imageFileName)),
                                      glob.glob('./test_images/st*.jpg')))


def imageSideBySide(leftImg, leftTitle, rightImg, rightTitle, figsize=(20, 10), leftCmap=None, rightCmap=None):
    """
    Display the images `leftImg` and `rightImg` side by side with image titles.
    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    if leftCmap == None:
        axes[0].imshow(leftImg)
    else:
        axes[0].imshow(leftImg, cmap=leftCmap)
    axes[0].set_title(leftTitle)

    if rightCmap == None:
        axes[1].imshow(rightImg)
    else:
        axes[1].imshow(rightImg, cmap=rightCmap)
    axes[1].set_title(rightTitle)
    fig.show()


imageSideBySide(
    cv2.cvtColor(test_images_straight_lines[0][1], cv2.COLOR_BGR2RGB), test_images_straight_lines[0][0],
    cv2.cvtColor(test_images_straight_lines[1][1], cv2.COLOR_BGR2RGB), test_images_straight_lines[1][0],
)


def show_comparison(image_1, image_2, title_1='original', title_2='image 2'):
    '''plot the two images passed to this function for comparison'''
    global original_image, fig, axes
    original_image = image_1
    image_with_corners = image_2
    fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
    axes[0].imshow(original_image)
    axes[0].set_title(title_1)
    axes[1].imshow(image_with_corners)
    axes[1].set_title(title_2)
    fig.show()


original_image = cv2.cvtColor(test_images_straight_lines[0][1], cv2.COLOR_BGR2RGB)
# correcting distortion in a image using camera matrix and coefficients
undistorted_image = cv2.undistort(original_image, mtx, dist, None, mtx)

xSize, ySize, _ = undistorted_image.shape
copy = undistorted_image.copy()

maximum_Y = 720
top_Y = 460

# here i draw the lines on both lanes for helping in visualization . The formation is trapezoidal
# first select points for left lane
left_bottom_x, left_bottom_y = (195, maximum_Y)
left_top_x, left_top_y = (590, top_Y)

# select points for right lanes
right_top_x, right_top_y = (700, top_Y)
right_bottom_x, right_bottom_y = (1125, maximum_Y)

color = [255, 0, 0]
# width for drawing lines
width = 4

# drawing trapezoid over lanes lines
cv2.line(copy, (left_bottom_x, left_bottom_y), (left_top_x, left_top_y), color, width)
cv2.line(copy, (left_top_x, left_top_y), (right_top_x, right_top_y), color, width)
cv2.line(copy, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), color, width)
cv2.line(copy, (right_bottom_x, right_bottom_y), (left_bottom_x, left_bottom_y), color, width)

fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(copy)
fig.show()

gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
# Here we going perform perspective transform of an image for which we take source and destination coordinates.
# The transformation done with the help of perspective transform matrix M. then we warped the image using
# linear interpolation
# source coordinates on original image
src_coordinates = np.float32([
    [left_top_x, left_top_y],
    [right_top_x, right_top_y],
    [right_bottom_x, right_bottom_y],
    [left_bottom_x, left_bottom_y]
])
n_X = gray.shape[1]
n_Y = gray.shape[0]
img_size = (n_X, n_Y)
offset = 200

# destination coordinates on image for transform
dst_coordinates = np.float32([
    [offset, 0],
    [img_size[0] - offset, 0],
    [img_size[0] - offset, img_size[1]],
    [offset, img_size[1]]
])
img_size = (gray.shape[1], gray.shape[0])

# calculating perspective transformation matrix
M = cv2.getPerspectiveTransform(src_coordinates, dst_coordinates)
# calculating the inverse of perspective transformation matrix
M_inverse = cv2.getPerspectiveTransform(dst_coordinates, src_coordinates)
# creating the warped image by linear interpolation
warped = cv2.warpPerspective(undistorted_image, M, img_size)

# display the original and warped image
show_comparison(original_image, warped, title_2="warped")

# serialize and save the perspective data in file for further use
pickle.dump({'M': M, 'Minv': M_inverse}, open('serialized_camera_data/perspective_transform.p', 'wb'))

# print the perspective parameters
print("The perspective Matrix :-\n", M)
print("The inverse of perspective Matrix :-\n", M_inverse)
