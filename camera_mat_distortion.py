import numpy as np
import pickle
import cv2
import glob
import matplotlib.pyplot as plt

# Get all images from directory in form of 'image name' & their matrix representation
images_with_names_for_calibration = list(
    map(lambda image_name: (image_name, cv2.imread(image_name)), glob.glob('./camera_cal/c*.jpg')))


def display_images(images, cols=4, rows=5, figsize=(15, 10), cmap=None):
    """
    This function display images for a quick look
    """
    no_of_images = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(cols * rows)
    for ax, index in zip(axes.flat, indexes):
        if index < no_of_images:
            imagePathName, image = images[index]
            if cmap == None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap)
            ax.set_title(imagePathName)
            ax.axis('off')
    fig.show()


display_images(images_with_names_for_calibration, 4, 5, (15, 13))


def generate_img_points():
    '''This Fn find & draws the corners on chess board images'''
    objpoints = []  # 3d points in real world space
    imgpoints = []  # image points on image
    output_images = []
    original_images = []
    # Here we created a matrix of 3d coordinates as image is flat we take Z coordinate 0
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    for image_with_names in images_with_names_for_calibration:
        file_name, image = image_with_names
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            corners_drawn_on_image = cv2.drawChessboardCorners(image.copy(), (9, 6), corners, ret)
            output_images.append(corners_drawn_on_image)
            original_images.append(image)
    return objpoints, imgpoints, output_images, original_images


objpoints, imgpoints, output_images, original_images = generate_img_points()

# displaying the exracted points on chessboard
# for index in range(0, 19):


index = 10
original_image = original_images[index]
image_with_corners = output_images[index]
fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
axes[0].imshow(original_image)
axes[0].set_title('Original')
axes[1].imshow(image_with_corners)
axes[1].set_title('Corners drawn')
fig.show()

# calibrating camera using image & object points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, original_image.shape[0:2], None, None)

undistorted_image = cv2.undistort(original_image, mtx, dist, None, mtx)

fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
axes[0].imshow(original_image)
axes[0].set_title('Original')
axes[1].imshow(undistorted_image)
axes[1].set_title('Undistorted')
fig.show()

# saving the transformation & coefficients
pickle.dump({'mtx': mtx, 'dist': dist}, open('serialized_camera_data/camera_calibration.p', 'wb'))
