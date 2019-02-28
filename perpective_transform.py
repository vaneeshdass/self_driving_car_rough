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
original = cv2.cvtColor(testImages[index][1], cv2.COLOR_BGR2RGB)
undist = cv2.undistort(original, mtx, dist, None, mtx)

xSize, ySize, _ = undist.shape
copy = undist.copy()

bottomY = 720
topY = 455

# drawing lines for both sides left and right lane to draw a rectangle for visualization
left1 = (190, bottomY)
left1_x, left1_y = left1
left2 = (585, topY)
left2_x, left2_y = left2

right1 = (705, topY)
right1_x, right1_y = right1

right2 = (1130, bottomY)
right2_x, right2_y = right2

color = [255, 0, 0]
w = 2
cv2.line(copy, left1, left2, color, w)
cv2.line(copy, left2, right1, color, w)
cv2.line(copy, right1, right2, color, w)
cv2.line(copy, right2, left1, color, w)
fig, ax = plt.subplots(figsize=(20, 10))
ax.imshow(copy)
fig.show()

gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
# source coordinates on original image
src = np.float32([
    [left2_x, left2_y],
    [right1_x, right1_y],
    [right2_x, right2_y],
    [left1_x, left1_y]
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
warped = cv2.warpPerspective(undist, M, img_size)

imageSideBySide(
    original, 'Original',
    warped, 'Perspective transformed'
)

pickle.dump({'M': M, 'Minv': Minv}, open('serialized_camera_data/perspective_transform.p', 'wb'))

print(M)
print(Minv)
