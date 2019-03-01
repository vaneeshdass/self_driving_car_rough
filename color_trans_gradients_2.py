import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


calibration_parameters = pickle.load(open('serialized_camera_data/camera_calibration.p', 'rb'))
mtx, dist = map(calibration_parameters.get, ('mtx', 'dist'))
def display_images(images, cols=4, rows=5, figsize=(15, 10), cmap=None):
    """
    This function display images for a quick look
    """
    no_of_images = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(cols * rows)
    for ax, index in zip(axes.flat, indexes):
        if index < no_of_images:
            image_name, image = images[index]
            if cmap == None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap)
            ax.set_title(image_name)
            ax.axis('off')
    fig.show()


def create_undistorted_hls_image(image, mtx=mtx, dist=dist):
    """
    This Fn first undistort the image and then returns the hls representation of the same.
    """
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    return cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HLS)


def delegator(images, function_pointer, display_image=True, cmap='gray'):
    """A utility function it just delegates the work to another function.
    retuns the list of tuples containing image name & respective image matrix.
    """
    result = list(map(lambda img: (img[0], function_pointer(img[1])), images))
    if display_image:
        display_images(result, 2, 3, (15, 13), cmap='gray')
    return result


def create_thresholded_binary_image(img, thresh_min, thresh_max):
    """
   This function apply the thresholds as given and convert the image in binary
    """
    xbinary = np.zeros_like(img)
    xbinary[(img >= thresh_min) & (img <= thresh_max)] = 1
    return xbinary


def create_sobel_image(img, direction='x', kernel_size=3, thresh_min=0, thresh_max=255):
    """
    This Fn apply Sobel gradient in both x or y direction as given to it & then create thresholded binary image of the
     same.
    """
    if direction == 'x':
        y = 0
        x = 1
    else:
        y = 1
        x = 0

    sobel_image = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=kernel_size)
    # calculating absolute value
    abs_sobel = np.absolute(sobel_image)
    scaled_sobel_image = np.uint8(255.0 * abs_sobel / np.max(abs_sobel))

    return create_thresholded_binary_image(scaled_sobel_image, thresh_min, thresh_max)


def calculate_magnitude_gradient(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
    """
  This function calculates the magnitude gradient over sobel & gives back binary image.
    """
    sobel_X = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_Y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # calculating gradient magnitude
    magnitude_gradient = np.sqrt(sobel_X ** 2 + sobel_Y ** 2)
    scaled = np.max(magnitude_gradient) / 255
    magnitude_gradient = (magnitude_gradient / scaled).astype(np.uint8)
    return create_thresholded_binary_image(magnitude_gradient, thresh_min, thresh_max)


def calculate_direction_gradient(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi / 2):
    """
    This function returns the binary image after calculating the direction magnitude
    """
    sobel_X = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_Y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absolute_direction_gradient = np.arctan2(np.absolute(sobel_Y), np.absolute(sobel_X))

    return create_thresholded_binary_image(absolute_direction_gradient, thresh_min, thresh_max)
