---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image01]: ./writeup/0.1.calibration.png "Calibration1"
[image02]: ./writeup/0.2.calibration.png "Calibration2"
[image03]: ./writeup/0.3.calibration.png "Calibration3"
[image1]: ./writeup/1.unwarped.png "Unwarped"
[image2]: ./writeup/2.undistorted.png "Undistorted"
[image3]: ./writeup/3.perspective.png "Perspective"
[image4]: ./writeup/3.color.png "Color"
[image5]: ./writeup/3.gradient.png "Gradient"
[image6]: ./writeup/4.perspective_binary.png "Binary"
[image7]: ./writeup/5.findinglanes.png "Histogram"
[image8]: ./writeup/6.lanesBlindSearch.png "Blind"
[image9]: ./writeup/7.lanesTargetedSearch.png "Targeted"
[image10]: ./writeup/8.curveMeasurement.png "Measurement"
[image11]: ./writeup/9.drawFinalLines.png "Final"
---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to calibrate the camera can be found in the method readCalibrationPoints at utils.py.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Also assuming the calibration images have 9 columns per 6 rows.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

These are some images used for calibration:

![alt text][image01]
![alt text][image02]
![alt text][image03]

After obtaining the calibration points, both `imgpoints` and `objpoints`, I saved them in a pickle file for future use.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function, in the method `calibrateCamera()` at `utils.py`. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warpPerspective()`, which appears in the method `perspectiveTransform()` at `utils.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src_frustrum = np.float32([[172,720], [586,450], [690,450], [1160,720]])
dst_frustrum = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 172, 720      | 320, 720      | 
| 586, 450      | 320, 0        |
| 690, 450      | 960, 0        |
| 1160, 720     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.

For the color filtering I used two channels to be able to detect white and yellow lane lines.
* HLS L Channel, focused on white lines
* LAB B Channel, focused on  yellow lines.

Here's an example of my output for this step. (note: this is not actually from one of the test images)

![alt text][image4]

The code for this can be found in the method `color_filter()` at `utils.py`.

Second step is to apply a Sobel X Gradient (code to be found at `sobelx_binary()` at `utils.py`) to reinforce vertical lines.

![alt text][image5]

Both color and gradient binaries are merged together with an OR logic operation to create my binary image. Final result:

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Now that we have a decent binary image, we will try to find the lanes. We will use the methods shown in class: Implement Sliding Windows and Fit a Polynomial, fit my lane lines with a 2nd order polynomial.

First, calculate the histogram of the binary image.

![alt text][image7]

A. Blind search.
For detecting the lane lines, we will use the method "Sliding Window Polyfit", starting from the bottom of the image to the top, dividing the image into 9 regions.
Code at `laneLinesBlindSearch` at `utils.py`.

![alt text][image8]

B. Targeted search.
This method will use information from the previous frame in order to speed up the calculations.
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
Code at `laneLinesTargetedSearch` at `utils.py`.

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I'm assuming:

* the camera is mounted at the center of the car.
* lanes of 30 meters long and 3.7 meters wide (around 700px in our images).

To calculate the radius we make use of the 2nd order polynomial calculated in the previous step for both left and right lines.
To calculate the offset frmo the center we calculate the distance of left and right line from the middle of the image.

![alt text][image10]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This final step draws everything back onto the original frame. Code available at `drawFinalLines()` at  `utils.py`.  Here is an example of my result on a test image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems I encountered:
* It´s very difficult to find tune the filters (color, gradient) that works in all type of road conditions, lighting enviroment, shadows, etc. (I spent too much time on this actually).
* I didn't take into account lane changing or different road line markings: like arrows, road works, crossings, etc.

Possible improvements:
* Apply different filters depending on the light and image conditions in the frame.
* Apply different filters in regions of the frame to be able to differentiate changes in road conditions or lighting.
