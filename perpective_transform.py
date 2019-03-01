import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# Loading camera calibration
cameraCalibration = pickle.load(open('serialized_camera_data/camera_calibration.p', 'rb'))
mtx, dist = map(cameraCalibration.get, ('mtx', 'dist'))

# Load calibration images.
testImages = list(map(lambda imageFileName: (imageFileName, cv2.imread(imageFileName)),
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
    cv2.cvtColor(testImages[0][1], cv2.COLOR_BGR2RGB), testImages[0][0],
    cv2.cvtColor(testImages[1][1], cv2.COLOR_BGR2RGB), testImages[1][0],
)

index = 0
original_image = cv2.cvtColor(testImages[index][1], cv2.COLOR_BGR2RGB)
# coreecting distortion in a image using transformation matrix and coefficients
undistorted_image = cv2.undistort(original_image, mtx, dist, None, mtx)

xSize, ySize, _ = undistorted_image.shape
copy = undistorted_image.copy()

maximum_Y = 720
top_Y = 455

# here i draw the lines on both lanes for helping in visualization . The formation is trapezoidal
#first select points for left lane
left1 = (190, maximum_Y)
left_bottom_x, left_bottom_y = (190, maximum_Y)
left2 = (585, top_Y)
left_top_x, left_top_y = (585, top_Y)

right1 = (705, top_Y)
right_top_x, right_top_y = right1

right2 = (1130, maximum_Y)
right_bottom_x, right_bottom_y = (1130, maximum_Y)

color = [255, 0, 0]
width = 2

cv2.line(copy, (left_bottom_x, left_bottom_y), (left_top_x, left_top_y), color, width)
cv2.line(copy, (left_top_x, left_top_y), (right_top_x, right_top_y), color, width)
cv2.line(copy, (right_top_x, right_top_y), (right_bottom_x, right_bottom_y), color, width)
cv2.line(copy, (right_bottom_x, right_bottom_y), (left_bottom_x, left_bottom_y), color, width)

fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(copy)
fig.show()

gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
# source coordinates on original image
src = np.float32([
    [left_top_x, left_top_y],
    [right_top_x, right_top_y],
    [right_bottom_x, right_bottom_y],
    [left_bottom_x, left_bottom_y]
])
nX = gray.shape[1]
nY = gray.shape[0]
img_size = (nX, nY)
offset = 200

# destination coordinates
dst = np.float32([
    [offset, 0],
    [img_size[0] - offset, 0],
    [img_size[0] - offset, img_size[1]],
    [offset, img_size[1]]
])
img_size = (gray.shape[1], gray.shape[0])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(undistorted_image, M, img_size)

imageSideBySide(
    original_image, 'Original',
    warped, 'Perspective transformed'
)

pickle.dump({'M': M, 'Minv': Minv}, open('serialized_camera_data/perspective_transform.p', 'wb'))

print(M)
print(Minv)
