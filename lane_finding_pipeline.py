import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


class LaneFindingPipeline:
    def __init__(self):
        self.mtx, self.dist = self.get_distortion_coefficients()

    def get_distortion_coefficients(self):
        '''
        Calibrate camera by getting the distortion coefficients from test images
        :return: camera matrix and distortion coefficients
        '''

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('camera_cal/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return mtx, dist

    def undistort_image(self, img, mtx, dist):
        '''
        Given a distorted image and a camera matrix and distortion coefficients, undistort the image
        :return: undistorted image
        '''
        img = np.copy(img)
        return cv2.undistort(img, mtx, dist, None, mtx)

    def colour_and_gradient_threshold(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        '''
        Apply colour and gradient thresholding to an input image.
        :param img: input image
        :param s_thresh: colour threshold
        :param sx_thresh: gradient theshold
        :return: thresholded image
        '''
        img = np.copy(img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:, :, 1]
        s_channel = hsv[:, :, 2]
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

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def perspective_transform(self, img, src_points, dst_points):
        '''
        Apply a perspective transform to the input image
        :param img: input image
        :param src_points: source points
        :param dst_points: destination points
        :return: transform matrix and transformed image.
        '''
        img = np.copy(img)
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return M, cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    def line_fitting_initial(self, img):
        '''
        Given a thresholded and transformed image, fit a polynomial to the lane line using the sliding window method.
        :param img: transformed and thresholded image
        :return: lane line information
        '''
        binary_warped = np.copy(img)
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        # Guess of left lane base
        leftx_base = np.argmax(histogram[:midpoint])
        # Guess of right lane base
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window, starting with peaks of histogram
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify shape of both windows
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within each window (i.e. potential lanes)
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append nonzero pixels to list
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If there are enough non-zero pixels in the window, shift the center of the window to the mean of the
            # non-zero pixels
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

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # plot found lines for debugging purposes
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        y_eval = np.max(ploty)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])

        return ploty, left_fitx, right_fitx, left_curverad, right_curverad

    def line_fitting_secondary(self, img, left_fit, right_fit):
        '''
        More efficient sliding window method that can be used after line_fitting_initial
        :param img: thresholded and transformed image
        :param left_fit: left lane
        :param right_fit: right lane
        :return: lane line information
        '''
        binary_warped = np.copy(img)
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        y_eval = np.max(ploty)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])

        return ploty, left_fitx, right_fitx, left_curverad, right_curverad

    def inverse_warp(self, img, warped, left_fitx, right_fitx, ploty, Minv):
        '''
        When we are done finding lanes in the transformed space, we want to view them in the original space.
        :param img: image in original space
        :param warped: image in transformed space
        :param left_fitx: left lane
        :param right_fitx: right lane
        :param Minv: Transformation matrix.
        :return: image in original space with lanes drawn on
        '''
        image = np.copy(img)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

    def apply_pipeline(self, img):
        '''
        Apply each step of the pipeline to a lane image
        '''

        img = np.copy(img)

        # source and destination points for the perspective transform step
        src = np.float32([[554, 480], [733, 480], [298, 654], [1000, 654]])
        dst = np.float32([[265, 95], [945, 95], [265, 576], [1105, 576]])

        # Step 1: Undistort image
        undistorted = self.undistort_image(img, self.mtx, self.dist)
        # Step 2: Apply colour and gradient thresholding
        thresholded = self.colour_and_gradient_threshold(img=undistorted)
        # Step 3: Apply a perspective transform for a top-down view
        M, warped = self.perspective_transform(thresholded, src, dst)
        # Step 3.5: We also need the inverse transform to get back to the original image space
        Minv, _ = self.perspective_transform(thresholded, dst, src)
        # Step 4: Fit polynomial line to the transform image
        ploty, left_fitx, right_fitx, left_curverad, right_curverad = self.line_fitting_initial(warped)
        # Step 5: Invert the transformation and return the image in the original space with the found lanes
        lane_img = self.inverse_warp(img, warped, left_fitx, right_fitx, ploty, Minv)
        return lane_img

    def find_lines_in_video(self, video_path):
        '''
        Overlay detected lanes in a video.
        '''
        clip = VideoFileClip(video_path)
        clip_with_lines = clip.fl_image(self.apply_pipeline)
        clip_with_lines.write_videofile('harder_challenge_video_with_lines.mp4', audio=False)


def main():
    test_undistort_image = cv2.imread('test_images/test6.jpg')
    lfp = LaneFindingPipeline()
    res = lfp.apply_pipeline(test_undistort_image)
    # plt.imshow(res)
    # plt.show()
    # vid = 'harder_challenge_video.mp4'
    # lfp.find_lines_in_video(vid)


if __name__ == "__main__":
    main()
