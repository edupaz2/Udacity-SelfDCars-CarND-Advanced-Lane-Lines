import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from utils import calibrateCamera, perspectiveTransform

from os import listdir

###
# Test spaces colors and best color channels to find the lanes
# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
# https://docs.opencv.org/3.4.1/de/d25/imgproc_color_conversions.html
###

# Camera calibration
mtx, dist = calibrateCamera()

# Test all images in test dir
if True == True:
	for imgname in listdir('test_images'):
		img = cv2.imread('test_images/' + imgname)
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_warped, perspective_M = perspectiveTransform(img, mtx, dist)
		rgb = cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB)
		hsv = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HSV)
		hls = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS)
		ycrcb = cv2.cvtColor(img_warped, cv2.COLOR_BGR2YCrCb)
		lab = cv2.cvtColor(img_warped, cv2.COLOR_BGR2LAB)

		# For whites
		hls_l_channel = (hls[:,:,1] > 203) & (hls[:,:,1] < 255)
		bgr_b_channel = (img_warped[:,:,1] > 203) & (img_warped[:,:,1] < 255)
		# For yellows
		lab_b_channel = (lab[:,:,2] > 140) & (lab[:,:,2] < 180)
		ycrcb_cr_channel = (ycrcb[:,:,1] > 93) & (ycrcb[:,:,1] < 135)
		ycrcb_cb_channel = (ycrcb[:,:,2] > 110) & (ycrcb[:,:,2] < 190)
		# Total

		cols = 3
		f, ax = plt.subplots(6, cols, figsize=(20, 16))
		ax = ax.ravel()
		ax[0*cols+0].imshow(rgb)
		ax[0*cols+0].set_title('Orig', fontsize=15)
		ax[0*cols+1].imshow(hls_l_channel, cmap='gray')
		ax[0*cols+1].set_title('Test1', fontsize=15)
		ax[0*cols+2].imshow(bgr_b_channel, cmap='gray')
		ax[0*cols+2].set_title('Test2', fontsize=15)
		##
		ax[1*cols+0].imshow(rgb[:,:,0], cmap='gray')
		ax[1*cols+0].set_title('R', fontsize=15)
		ax[1*cols+1].imshow(rgb[:,:,1], cmap='gray')
		ax[1*cols+1].set_title('G', fontsize=15)
		ax[1*cols+2].imshow(rgb[:,:,2], cmap='gray')
		ax[1*cols+2].set_title('B', fontsize=15)
		##
		ax[2*cols+0].imshow(hsv[:,:,0], cmap='gray')
		ax[2*cols+0].set_title('H', fontsize=15)
		ax[2*cols+1].imshow(hsv[:,:,1], cmap='gray')
		ax[2*cols+1].set_title('S', fontsize=15)
		ax[2*cols+2].imshow(hsv[:,:,2], cmap='gray')
		ax[2*cols+2].set_title('V', fontsize=15)
		##
		ax[3*cols+0].imshow(hls[:,:,0], cmap='gray')
		ax[3*cols+0].set_title('H', fontsize=15)
		ax[3*cols+1].imshow(hls[:,:,1], cmap='gray')
		ax[3*cols+1].set_title('L', fontsize=15)
		ax[3*cols+2].imshow(hls[:,:,2], cmap='gray')
		ax[3*cols+2].set_title('S', fontsize=15)
		##
		ax[4*cols+0].imshow(ycrcb[:,:,0], cmap='gray')
		ax[4*cols+0].set_title('Y', fontsize=15)
		ax[4*cols+1].imshow(ycrcb[:,:,1], cmap='gray')
		ax[4*cols+1].set_title('Cr', fontsize=15)
		ax[4*cols+2].imshow(ycrcb[:,:,2], cmap='gray')
		ax[4*cols+2].set_title('Cb', fontsize=15)
		##
		ax[5*cols+0].imshow(lab[:,:,0], cmap='gray')
		ax[5*cols+0].set_title('L', fontsize=15)
		ax[5*cols+1].imshow(lab[:,:,1], cmap='gray')
		ax[5*cols+1].set_title('A', fontsize=15)
		ax[5*cols+2].imshow(lab[:,:,2], cmap='gray')
		ax[5*cols+2].set_title('B', fontsize=15)
		##
		plt.suptitle(imgname)
		plt.show()

# After analysis, we choose HSL S-channel for whites, and LAB B-channel for yellows.
if True == True:
	for imgname in listdir('test_images'):
		img = cv2.imread('test_images/' + imgname)
		img_warped, perspective_M = perspectiveTransform(img, mtx, dist)
		hls = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS)
		lab = cv2.cvtColor(img_warped, cv2.COLOR_BGR2LAB)
		ycrcb = cv2.cvtColor(img_warped, cv2.COLOR_BGR2YCrCb)

		# For whites
		hls_l_channel = (hls[:,:,1] > 203) & (hls[:,:,1] < 255)
		bgr_b_channel = (img_warped[:,:,1] > 203) & (img_warped[:,:,1] < 255)
		# For yellows
		lab_b_channel = (lab[:,:,2] > 140) & (lab[:,:,2] < 180)
		ycrcb_cr_channel = (ycrcb[:,:,1] > 93) & (ycrcb[:,:,1] < 135)
		ycrcb_cb_channel = (ycrcb[:,:,2] > 110) & (ycrcb[:,:,2] < 190)

		#displayImagesSplitInChannels(imgname, [[img, 'BGR'], [color_thresholds, 'mask']])
		f, ax = plt.subplots(3, 2, figsize=(16, 12))
		ax = ax.ravel()
		ax[0].imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
		ax[1].imshow((hls_l_channel&bgr_b_channel) ^ ((~ycrcb_cr_channel)&(~ycrcb_cb_channel)), cmap='gray')
		
		ax[2].imshow(hls_l_channel, cmap='gray')
		ax[3].imshow(bgr_b_channel, cmap='gray')

		#ax[4].imshow(lab_b_channel, cmap='gray')
		ax[4].imshow(~(ycrcb_cr_channel), cmap='gray')
		ax[5].imshow(~(ycrcb_cb_channel), cmap='gray')
		plt.suptitle(imgname)
		plt.show()