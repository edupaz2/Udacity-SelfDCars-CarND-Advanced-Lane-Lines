from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

CALIBRATION_IMAGES_PATH = 'camera_cal/calibration*.jpg'
CALIBRATION_POINTS_PATH = 'camera_cal/calibration.p'
nx = 9
ny = 6
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def displayImages(imglist):
	# imglist items will have the format [img, label, cmap]
	count = len(imglist)
	f, (ax) = plt.subplots(1, count, figsize=(10, 4))
	f.tight_layout()
	ax = ax.ravel()
	for i in range(count):
		ax[i].imshow(imglist[i][0], cmap=imglist[i][2])
		ax[i].set_title(imglist[i][1], fontsize=35)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()

def readCalibrationPoints():
	imgpoints = []
	objpoints = []
	# read existing calibration
	try:
		with open(CALIBRATION_POINTS_PATH, 'rb') as pfile:
			print('Reading calibration file', CALIBRATION_POINTS_PATH)
			cal_pickle = pickle.load(pfile)
			objpoints = cal_pickle["objpoints"]
			imgpoints = cal_pickle["imgpoints"]
	except Exception as e:
		print('Unable to read data from', CALIBRATION_POINTS_PATH, ':', e)

	if len(imgpoints) and len(objpoints):
		return imgpoints,objpoints

	# If there is no previous calibration, do it.
	print('Calibrating camera')
	# Do the calibration
	# prepare object points
	objp = np.zeros((nx*ny, 3), np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
	# Read camera_cal files
	images = glob(CALIBRATION_IMAGES_PATH)
	gray= None
	for fname in images:
		img = cv2.imread(fname)
		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		# If found, draw corners
		if ret == True:
			print('Corners found at', fname)
			# Append the results
			imgpoints.append(corners)
			objpoints.append(objp)

	# save to file
	try:
		with open(CALIBRATION_POINTS_PATH, 'wb+') as pfile:
			print('Saving to calibration file', CALIBRATION_POINTS_PATH)
			pickle.dump(
			{
				'imgpoints': imgpoints,
				'objpoints': objpoints,
			},
			pfile, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', CALIBRATION_POINTS_PATH, ':', e)

	return imgpoints,objpoints

def calibrateCamera(debug=False):
	imgpoints, objpoints = readCalibrationPoints()

	# Read in an image
	img = cv2.imread('camera_cal/calibration2.jpg')
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
	
	return mtx, dist

#src_frustrum = np.float32([[186,720], [606,441], [672,441], [1125,720]])
src_frustrum = np.float32([[172,720], [586,450], [690,450], [1160,720]])
dst_frustrum = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])
def distortionCorrection(img, mtx, dist, debug=False):
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	if debug == True:
		 displayImages( [[img, 'Original', None], [undist, 'Undistorted', None]] )
	return undist

def perspectiveTransform(img, src=src_frustrum, dst=dst_frustrum, debug=False):
	img_size = (img.shape[1], img.shape[0])
	M = cv2.getPerspectiveTransform(src, dst)
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	if debug == True:
		 displayImages( [[img, 'Original', None], [warped, 'Perspective', None]] )
	return warped, M

def color_filter(img):
	# Separate the chosen channels
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	hls_l_channel = hls[:,:,1]
	lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
	lab_b_channel = lab[:,:,2]

	hls_l_thresh=(210, 255)
	lab_b_thres=(192, 255)

    # Threshold color channels
	l_binary = np.zeros_like(hls_l_channel)
	l_binary[(hls_l_channel >= hls_l_thresh[0]) & (hls_l_channel <= hls_l_thresh[1])] = 1
	b_binary = np.zeros_like(lab_b_channel)
	b_binary[(lab_b_channel > lab_b_thres[0]) & (lab_b_channel < lab_b_thres[1])] = 1

	img_filtered = np.zeros_like(img[:,:,0])
	img_filtered[ (l_binary == 1) | (b_binary == 1) ] = 1

	return img_filtered

def sobelx_binary(img, lx_thresh=(25, 60)):
    # Separate the chosen channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    hls_s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(hls_s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    lxbinary = np.zeros_like(scaled_sobel)
    lxbinary[(scaled_sobel >= lx_thresh[0]) & (scaled_sobel <= lx_thresh[1])] = 1

    return lxbinary

def get_binary_image(img):
    color_binary = color_filter(img)
    lxbinary = sobelx_binary(img)

    combined_binary = np.zeros_like(lxbinary)
    combined_binary[(color_binary == 1) | (lxbinary == 1)] = 1
    
    return combined_binary

def laneLinesBlindSearch(binary_warped, debug=False):

	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]//2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 10
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]//nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
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
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
		(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
		(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
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

	# Fit a second order polynomial to each
	print(len(lefty), len(leftx), len(righty), len(rightx))
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	if debug == True:
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		# plt.imshow(out_img)
		# plt.plot(left_fitx, ploty, color='yellow')
		# plt.plot(right_fitx, ploty, color='yellow')
		# plt.xlim(0, binary_warped.shape[1])
		# plt.ylim(binary_warped.shape[0], 0)
		# plt.show()

		return left_fit, right_fit, out_img
	
	return left_fit, right_fit, None

def laneLinesTargetedSearch(binary_warped, left_fit, right_fit, debug=False):
	# Assume you now have a new warped binary image 
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin)))

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	if debug == True:
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
		                              ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
		                              ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
		# plt.imshow(result)
		# plt.plot(left_fitx, ploty, color='yellow')
		# plt.plot(right_fitx, ploty, color='yellow')
		# plt.xlim(0, binary_warped.shape[1])
		# plt.ylim(binary_warped.shape[0], 0)
		# plt.show()

		return left_fit, right_fit, result

	return left_fit, right_fit, None

def curvatureAndOffsetMeasurement(binary_warped, left_fit, right_fit, debug=False):
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	# Plot up the data
	if debug == True:
		plt.xlim(0, binary_warped.shape[1])
		plt.ylim(binary_warped.shape[0], 0)
		plt.plot(left_fitx, ploty, color='green', linewidth=3)
		plt.plot(right_fitx, ploty, color='green', linewidth=3)
		plt.show()

	#######
	# Define y-value where we want radius of curvature
	# I'll choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)
	left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

	#######

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	middleImg = binary_warped.shape[1]/2
	middleLane = (right_fitx[-1]-left_fitx[-1])/2

	# Now our radius of curvature is in meters
	print('LeftCurve:', left_curverad, 'm. RightCurve:', right_curverad, 'm. Middle: ', middleLane)

	return (left_curverad+right_curverad)/2, (middleImg-middleLane)*xm_per_pix

def drawFinalLines(image, binary_warped, radius, offset, perspective_M, left_fit, right_fit, debug=False):
	##########
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  
	##########

	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the binary_warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

	##########
	#color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	#color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	##########

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	Minv = np.linalg.inv(perspective_M)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)

	# Write Text
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 2
	fontColor              = (255,255,255)
	lineType               = 3

	offsetStr = 'Offset from center: {0:.2f}m'.format(offset)
	result = cv2.putText(result,offsetStr, (100, 75), font, fontScale, fontColor, lineType)
	radiusStr = 'Radius of curve: {0:.2f}m'.format(radius)
	result = cv2.putText(result, radiusStr, (100, 150), font, fontScale, fontColor, lineType)

	if debug == True:
		plt.imshow(result)
		plt.show()

	return result