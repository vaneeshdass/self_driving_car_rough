from color_trans_gradients_2 import delegator, create_undistorted_hls_image, create_sobel_image

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# Loading camera calibration
cameraCalibration = pickle.load(open('serialized_camera_data/camera_calibration.p', 'rb'))
mtx, dist = map(cameraCalibration.get, ('mtx', 'dist'))

# Load test images.
test_images_with_names = list(map(lambda imageFileName: (imageFileName, cv2.imread(imageFileName)),
                                  glob.glob('./test_images/*.jpg')))

original_image = test_images_with_names[1][1]

hls_image = create_undistorted_hls_image(original_image)

create_saturation_channel_images = lambda img: create_undistorted_hls_image(img)[:, :, 2]

take_sobel_in_X = lambda img: create_sobel_image(create_saturation_channel_images(img), thresh_min=10, thresh_max=160)

take_sobel_in_Y = lambda img: create_sobel_image(create_saturation_channel_images(img), direction='y', thresh_min=10,
                                                 thresh_max=160)


def combine_sobel_gradients(img):
    """
    Here we calculate the sobel along x & y
    """
    sobel_X = take_sobel_in_X(img)
    sobel_Y = take_sobel_in_Y(img)
    combined_sobel = np.zeros_like(sobel_X)
    combined_sobel[((sobel_X == 1) & (sobel_Y == 1))] = 1
    return combined_sobel


combined_sobel_image = delegator(test_images_with_names, combine_sobel_gradients, display_image=False)

perspective_matrix = pickle.load(open('serialized_camera_data/perspective_transform.p', 'rb'))
M, Minv = map(perspective_matrix.get, ('M', 'Minv'))


def do_perspective_transformation(image, M=M):
    """
    Adjust the `image` using the transformation matrix `M`.
    """
    img_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size)
    return warped


do_combine_sobel_transform = lambda img: do_perspective_transformation(combine_sobel_gradients(img))

transformed_binary_images = delegator(test_images_with_names, do_combine_sobel_transform, display_image=False,
                                      cmap='gray')

# conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


def search_lanes_and_fit_polynomial(image, nwindows=9, margin=110, minpix=50):
    """
    This Function search the lane pixels & then try to fit the polynomial on both lanes.
     Returns (left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
    """
    # get a perpective transformed image
    binary_warped_image = do_combine_sobel_transform(image)
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped_image[binary_warped_image.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped_image, binary_warped_image, binary_warped_image)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped_image.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # iterate through the windows one by one as we have to cover whole lane
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        y_low_coordinate = binary_warped_image.shape[0] - (window + 1) * window_height
        y_high_coordinate = binary_warped_image.shape[0] - window * window_height
        x_left_low = leftx_current - margin
        x_left_high = leftx_current + margin
        x_right_low = rightx_current - margin
        x_right_high = rightx_current + margin

        # Draw the windows on the visualization image, this draw a rectangle(window) on each iteration
        cv2.rectangle(out_img, (x_left_low, y_low_coordinate), (x_left_high, y_high_coordinate), (0, 255, 0), 2)
        cv2.rectangle(out_img, (x_right_low, y_low_coordinate), (x_right_high, y_high_coordinate), (0, 255, 0), 2)

        # These are the coordinates which are inside our window
        good_left_inds = ((nonzeroy >= y_low_coordinate) & (nonzeroy < y_high_coordinate) & (nonzerox >= x_left_low) & (
                nonzerox < x_left_high)).nonzero()[0]
        good_right_inds = \
            ((nonzeroy >= y_low_coordinate) & (nonzeroy < y_high_coordinate) & (nonzerox >= x_right_low) & (
                    nonzerox < x_right_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each lane, this representation is in pixels
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # this representation is for real word
    # Fit a second order polynomial to each lane
    left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)


def draw_windows_and_fitted_lines(image, ax):
    """
    This method draws the windows and fitted line on each image with the help of 'search_lanes_and_fit_polynomial' Fn.
    Returns (`left_fit` and `right_fit`)
    """
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = search_lanes_and_fit_polynomial(
        image)
    # Visualization
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # color left lane with red
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # color right lane with blue
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    ax.imshow(out_img)
    # plotting the fitted curve
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    return (left_fit, right_fit, left_fit_m, right_fit_m)


def draw_lane_lines_on_all_images(images, cols=2, rows=3, figsize=(15, 13)):
    """
   This method calls draw_windows_and_fitted_lines Fn for each image and then show the grid of output images.
    """
    no_of_images = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(cols * rows)
    image_path_with_fitted_parameters = []
    for ax, index in zip(axes.flat, indexes):
        if index < no_of_images:
            image_path, image = images[index]
            left_fit, right_fit, left_fit_m, right_fit_m = draw_windows_and_fitted_lines(image, ax)
            ax.set_title(image_path)
            ax.axis('off')
            image_path_with_fitted_parameters.append((image_path, left_fit, right_fit, left_fit_m, right_fit_m))
    fig.show()

    return image_path_with_fitted_parameters


# Lets do some action and draw the polygon & windows on images
images_top_view_with_curve = draw_lane_lines_on_all_images(test_images_with_names)
print('done')


# here we calculating curvature
def find_radius_of_curvature(yRange, left_fit_cr):
    """
    This Fn finds & returns the curvature of the polynomial
    """

    return ((1 + (2 * left_fit_cr[0] * yRange * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])


for image_polygon in images_top_view_with_curve:
    image_path, left_fit, right_fit, left_fit_m, right_fit_m = image_polygon
    max_Y = 719
    # converting from meters to kilometers
    leftCurvature = find_radius_of_curvature(max_Y, left_fit_m) / 1000
    rightCurvature = find_radius_of_curvature(max_Y, right_fit_m) / 1000
    print('Image : {}, Left : {:.2f} km, Right : {:.2f} km'.format(image_path, leftCurvature, rightCurvature))


# Warp the lane boundaries on top of image

def fill_the_lane_area(img, left_fit, right_fit):
    """
    This Fn calculate the polynomial & fill the lanes area using fillPolly method.
    """
    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(img).astype(np.uint8)

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # format the points for fillPoly method
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill the lanes area onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank image back to original image space using inverse perspective matrix we calculated earlier.
    new_warped_image = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1, new_warped_image, 0.3, 0)


def draw_lanes_visualization(img):
    """
   This Fn search the lane lines and best polynomial fit and then fills the lane area with the help
   of 'fill_the_lane_area' Fn.
    """
    # calculating the polynomial parameters
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = search_lanes_and_fit_polynomial(
        img)
    result = fill_the_lane_area(img, left_fit, right_fit)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


images_with_lane_area = delegator(test_images_with_names, draw_lanes_visualization)

print('done')


# car position

def find_car_position_and_show_output(img, fontScale=2):
    """
    This Fn find the car position and reflect the output on image
    """
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = search_lanes_and_fit_polynomial(
        img)
    output = fill_the_lane_area(img, left_fit, right_fit)

    # Calculate curvature
    left_curvature = find_radius_of_curvature(max_Y, left_fit_m)
    right_curvature = find_radius_of_curvature(max_Y, right_fit_m)

    # find car position
    # maximum distance along x-axis in meters
    x_max_mtrs = img.shape[1] * xm_per_pix
    # maximum distance along y-axis in meters
    y_max_mtrs = img.shape[0] * ym_per_pix

    car_center_pt = x_max_mtrs / 2
    # left line position in meters from left
    left_line = left_fit_m[0] * y_max_mtrs ** 2 + left_fit_m[1] * y_max_mtrs + left_fit_m[2]
    # right line position in meters from left
    right_line = right_fit_m[0] * y_max_mtrs ** 2 + right_fit_m[1] * y_max_mtrs + right_fit_m[2]
    # middle_point on center which is a refrence point to calculate the car distance
    middle_line = left_line + (right_line - left_line) / 2

    # car position from center
    car_distance_from_center = middle_line - car_center_pt
    if car_distance_from_center > 0:
        message = '{:.2f} m right'.format(car_distance_from_center)
    else:
        message = '{:.2f} m left'.format(-car_distance_from_center)

    # Writing the text on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    cv2.putText(output, 'Left curvature: {:.0f} m'.format(left_curvature), (50, 50), font, fontScale, font_color, 2)
    cv2.putText(output, 'Right curvature: {:.0f} m'.format(right_curvature), (50, 120), font, fontScale, font_color, 2)
    cv2.putText(output, 'Car is {} of center'.format(message), (50, 190), font, fontScale, font_color, 2)
    return output


image_output = delegator(test_images_with_names,
                         lambda img: cv2.cvtColor(find_car_position_and_show_output(img), cv2.COLOR_BGR2RGB))

print('done')

# video pipeline

from moviepy.editor import VideoFileClip


class Lane():
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_fit_m = None
        self.right_fit_m = None
        self.left_curvature = None
        self.right_curvature = None


def calculate_car_distance_and_lane_parameters(img):
    """
    This Fn for video pipeline calculates the curvature and car distance from center
    """
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = search_lanes_and_fit_polynomial(
        img)
    # Calculate curvature
    left_curvature = find_radius_of_curvature(max_Y, left_fit_m)
    right_curvature = find_radius_of_curvature(max_Y, right_fit_m)

    # find car position
    # maximum distance along x-axis in meters
    x_max_mtrs = img.shape[1] * xm_per_pix
    # maximum distance along y-axis in meters
    y_max_mtrs = img.shape[0] * ym_per_pix

    car_center_pt = x_max_mtrs / 2
    left_line = left_fit_m[0] * y_max_mtrs ** 2 + left_fit_m[1] * y_max_mtrs + left_fit_m[2]
    right_line = right_fit_m[0] * y_max_mtrs ** 2 + right_fit_m[1] * y_max_mtrs + right_fit_m[2]
    middle_line = left_line + (right_line - left_line) / 2

    car_distance_from_center = middle_line - car_center_pt

    return (left_fit, right_fit, left_fit_m, right_fit_m, left_curvature, right_curvature, car_distance_from_center)


def show_output_on_video_frame(img, left_fit, right_fit, left_fit_m, right_fit_m, leftCurvature, rightCurvature,
                               diffFromVehicle):
    """
   This Fn display the curvature and car distance for video frames
    """
    output = fill_the_lane_area(img, left_fit, right_fit)

    if diffFromVehicle > 0:
        message = '{:.2f} m right'.format(diffFromVehicle)
    else:
        message = '{:.2f} m left'.format(-diffFromVehicle)

    # Draw info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    cv2.putText(output, 'Left curvature: {:.0f} m'.format(leftCurvature), (50, 50), font, 1, font_color, 2)
    cv2.putText(output, 'Right curvature: {:.0f} m'.format(rightCurvature), (50, 120), font, 1, font_color, 2)
    cv2.putText(output, 'Vehicle is {} of center'.format(message), (50, 190), font, 1, font_color, 2)
    return output


def process_video(input_video, output_video):
    """
    This Fn takes in the input video, process it frame by frame and draw the lanes.
    Then it calculate the curvature and car distance using helper Fns and finally show the data over on each
    frame of video.
    It return the processed video which contains lane and data drawn.
    """
    video_file = VideoFileClip(input_video).subclip(0, 7)

    left_lane = Lane()
    right_lane = Lane()

    # nested Function to process the each image frame
    def processImage(img):
        left_fit, right_fit, left_fit_m, right_fit_m, left_curvature, right_curvature, diffFromVehicle = calculate_car_distance_and_lane_parameters(
            img)

        # applying sanity check based on left curvature
        if left_curvature > 10000:
            left_fit = left_lane.left_fit
            left_fit_m = left_lane.left_fit_m
            left_curvature = left_lane.left_curvature
        else:
            left_lane.left_fit = left_fit
            left_lane.left_fit_m = left_fit_m
            left_lane.left_curvature = left_curvature

        # applying sanity check based on right curvature
        if right_curvature > 10000:
            right_fit = right_lane.right_fit
            right_fit_m = right_lane.right_fit_m
            right_curvature = right_lane.right_curvature
        else:
            right_lane.right_fit = right_fit
            right_lane.right_fit_m = right_fit_m
            right_lane.right_curvature = right_curvature

        return show_output_on_video_frame(img, left_fit, right_fit, left_fit_m, right_fit_m, left_curvature,
                                          right_curvature,
                                          diffFromVehicle)

    # saving the result video
    clip = video_file.fl_image(processImage)
    clip.write_videofile(output_video, audio=False)


# Project video
process_video('./challenge_video.mp4', './challenge_video_output.mp4')
