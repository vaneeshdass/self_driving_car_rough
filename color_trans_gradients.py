import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# getting camera calibration parameters from file
calibration_parameters = pickle.load(open('serialized_camera_data/camera_calibration.p', 'rb'))
mtx, dist = map(calibration_parameters.get, ('mtx', 'dist'))

# getting tet images
test_images_with_names = list(map(lambda imageFileName: (imageFileName, cv2.imread(imageFileName)),
                                  glob.glob('test_images/*.jpg')))


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





# showImages(list(map(lambda img: (img[0], cv2.cvtColor(img[1], cv2.COLOR_BGR2RGB)), testImages)), 2, 3, (15, 13))

# lets select a random image
original_image = test_images_with_names[2][1]


def create_undistorted_hls_image(image, mtx=mtx, dist=dist):
    """
    This Fn first undistort the image and then returns the hls representation of the same.
    """
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
    return cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HLS)


# create hls representation of an image
hls_image = create_undistorted_hls_image(original_image)
# show all the channels of image(Hue, Lightness & Saturation)
fig, axes = plt.subplots(ncols=3, figsize=(20, 10))
for index, a in enumerate(axes):
    a.imshow(hls_image[:, :, index], cmap='gray')
    a.axis('off')


# def applyAndPack(images, action):
#     """
#     Images is a colletion of pairs (`title`, image). This function applies `action` to the image part of `images`
#     and pack the pair again to form (`title`, `action`(image)).
#     """
#     return list(map(lambda img: (img[0], action(img[1])), images))

# Act as delegator and shows the images after the transform
def delegator(images, function_pointer):
    """A utility function it just delegates the work to another function.
    retuns the list of tuples containing image name & respective image matrix.
    """
    result = list(map(lambda img: (img[0], function_pointer(img[1])), images))
    display_images(result, 2, 3, (15, 13), cmap='gray')
    return result


create_saturation_channel_images = lambda img: create_undistorted_hls_image(img)[:, :, 2]

saturation_images = delegator(test_images_with_names, create_saturation_channel_images)


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


# pointer to take derivative in x direction
take_sobel_in_X = lambda img: create_sobel_image(create_saturation_channel_images(img), thresh_min=10, thresh_max=160)
# sobel images in x direction
sobel_images_X = delegator(test_images_with_names, take_sobel_in_X)

# pointer to take derivative in y direction
take_sobel_in_Y = lambda img: create_sobel_image(create_saturation_channel_images(img), direction='y', thresh_min=10,
                                                 thresh_max=160)
# sobel images in y direction
sobel_images_Y = delegator(test_images_with_names, take_sobel_in_Y)


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


take_magnitude_gradient = lambda img: calculate_magnitude_gradient(create_saturation_channel_images(img), thresh_min=5,
                                                                   thresh_max=160)

magnitude_gradient_images = delegator(test_images_with_names, take_magnitude_gradient)


def calculate_direction_gradient(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi / 2):
    """
    This function returns the binary image after calculating the direction magnitude
    """
    sobel_X = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_Y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absolute_direction_gradient = np.arctan2(np.absolute(sobel_Y), np.absolute(sobel_X))

    return create_thresholded_binary_image(absolute_direction_gradient, thresh_min, thresh_max)


take_direction_gradient = lambda img: calculate_direction_gradient(create_saturation_channel_images(img),
                                                                   thresh_min=0.79,
                                                                   thresh_max=1.20)

direction_gradient_images = delegator(test_images_with_names, take_direction_gradient)


def merge_all_gradients(img):
    """
     This function compute the combination of Sobel X and Sobel Y or Magnitude and Direction as a whole
    """
    # below calculating all the respecetive gradients
    sobel_X = take_sobel_in_X(img)
    sobel_Y = take_sobel_in_Y(img)

    magnitude_gradient = take_magnitude_gradient(img)
    direction_gradient = take_direction_gradient(img)
    combined_gradient = np.zeros_like(sobel_X)
    # combining all as a whole
    combined_gradient[((sobel_X == 1) & (sobel_Y == 1)) | ((magnitude_gradient == 1) & (direction_gradient == 1))] = 1

    return combined_gradient


# here we have combined images of all gradients
merged_images = delegator(test_images_with_names, merge_all_gradients)

# To display a summary of all the conversions
titles = ['Sobel-X', 'Sobel-Y', 'Magnitude', 'Direction', 'Combined']

images_with_gradients = list(zip(sobel_images_X, sobel_images_Y, magnitude_gradient_images, direction_gradient_images, merged_images))
#selected 3 images
image_with_names = list(map(lambda images: list(zip(titles, images)), images_with_gradients))[3:6]
flatten_results = [item for sublist in image_with_names for item in sublist]

fig, axes = plt.subplots(ncols=5, nrows=len(image_with_names), figsize=(25, 10))
for ax, imageTuple in zip(axes.flat, flatten_results):
    title, images = imageTuple
    imagePath, img = images
    ax.imshow(img, cmap='gray')
    ax.set_title(imagePath + '\n' + title, fontsize=8)
    ax.axis('off')
fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0)
fig.show()
