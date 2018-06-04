import utils
from line import Line
from os import listdir

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

DEBUG = False
PIPELINE_IMAGE = False
DUMP_VIDEO = False
DEBUG_VIDEO = False 

# Camera calibration
mtx, dist = utils.calibrateCamera(debug=DEBUG)

if PIPELINE_IMAGE == True:
    if mtx is not None:
        # Test Distortion correction
        img = cv2.imread('camera_cal/calibration1.jpg')
        utils.displayImages( [[img, 'Original', None], [cv2.undistort(img, mtx, dist, None, mtx), 'Undistorted', None]])


    for imgname in listdir('test_images'):
        # 0. Read the image
        img = cv2.imread('test_images/' + imgname)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Distortion correction
        undist = utils.distortionCorrection(RGB_img, mtx, dist, debug=DEBUG)

        # 2. Perspective transform
        img_warped, perspective_M = utils.perspectiveTransform(undist, debug=DEBUG)

        # 3. Color/gradient threshold
        binary_warped = utils.get_binary_image(img_warped)

        # Display the histogram
        if DEBUG == True:
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
            plt.plot(histogram)
            plt.show()

        # Decide which pixels are lane lines pixels.
        # 5. Locate the lane lines, blind search
        left_fit, right_fit = utils.laneLinesBlindSearch(binary_warped, debug=DEBUG)

        # 6. Locate the lane lines, targeted search
        left_fit, right_fit = utils.laneLinesTargetedSearch(binary_warped, left_fit, right_fit, debug=DEBUG)

        # 7. Measuring the curvature
        radius, offset = utils.curvatureAndOffsetMeasurement(binary_warped, left_fit, right_fit, debug=DEBUG)

        utils.drawFinalLines(RGB_img, binary_warped, radius, offset, perspective_M, left_fit, right_fit, debug=DEBUG)

if DUMP_VIDEO == True:
    video_name = 'project_video'
    #video_name = 'challenge_video'
    #video_name = 'harder_challenge_video'
    clip = VideoFileClip(video_name + '.mp4')

    saveImgCount = 0
    for frame in clip.iter_frames():
        imgname = 'test_images/{0}_frame{1}.jpg'.format(video_name, saveImgCount)
        cv2.imwrite(imgname, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        saveImgCount = saveImgCount + 1

def process_image_pipeline(image):
    # image comes in BGR format
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undist = utils.distortionCorrection(RGB_img, mtx, dist)
    img_warped, perspective_M = utils.perspectiveTransform(undist)
    binary_img = utils.get_binary_image(img_warped)
    if not left_line.detected or not right_line.detected:
        left_fit, right_fit, searchImg = utils.laneLinesBlindSearch(binary_img, debug=False)
        left_line.add_fit(left_fit)
        right_line.add_fit(right_fit)
    else:
        left_fit, right_fit, searchImg = utils.laneLinesTargetedSearch(binary_img, left_line.best_fit, right_line.best_fit, debug=False)
        left_line.add_fit(left_fit)
        right_line.add_fit(right_fit)
    
    radius = (left_line.radius_of_curvature+right_line.radius_of_curvature)/2
    middleImg = binary_img.shape[1]/2
    middleLane = (left_line.line_base_pos + right_line.line_base_pos)/2
    offset = (middleImg-middleLane)*utils.xm_per_pix
    result = utils.drawFinalLines(image, binary_img, radius, offset, perspective_M, left_line.best_fit, right_line.best_fit)
    print('LeftCurve:', left_line.radius_of_curvature, 'm. RightCurve:', right_line.radius_of_curvature, 'm. Middle: ', middleLane)
    
    if DEBUG_VIDEO == True:
        # Write Text
        if left_line.detected:
            result = cv2.putText(result, "L", (50, 575), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 4)
        else:
            result = cv2.putText(result, "L", (50, 575), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 4)

        if right_line.detected:
            result = cv2.putText(result, "R", (1200, 575), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 4)
        else:
            result = cv2.putText(result, "R", (1200, 575), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 4)


        fig, ax = plt.subplots(2, 2, figsize=(16, 12))
        ax = ax.ravel()
        ax[0].imshow(image)
        ax[1].imshow(img_warped, cmap='gray')
        ax[2].imshow(binary_img, cmap='gray')
        #ax[3].imshow(searchImg, cmap='gray')
        ax[3].imshow(result)
        #ax[5].imshow(result)

        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    return result

video_name = 'project_video.mp4'
#video_name = 'challenge_video.mp4'
#video_name = 'harder_challenge_video.mp4'
if DEBUG_VIDEO == True:
    video_output = 'result.debug.' + video_name
    
    left_line = Line()
    right_line = Line()
    
    clip = VideoFileClip(video_name)
    output_clip = clip.fl_image(process_image_pipeline)
    output_clip.write_videofile(video_output, audio=False)

video_output = 'result.' + video_name

left_line = Line()
right_line = Line()
    
clip = VideoFileClip(video_name)
output_clip = clip.fl_image(process_image_pipeline)
output_clip.write_videofile(video_output, audio=False)
