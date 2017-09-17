# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[pipeline]: ./writeup_images/pipeline.png "Example pipeline"

### 1. Pipeline description

The pipeline consisted of this next steps:

* Convert the original image to the HSL space; this was done in order to successfully detect the lanes in the challenge video. This color space allowed me to detect the yellow left lane on the part of the video where the highway become a very light color;
* Convert the HSL image to grayscale (in preparation for Canny edge detection)
* Blur the grayscale image with a Gaussian kernel of 11
* Apply the Canny edge detection algorithm
* After cropping a trapezoid region of interest, I applied the Hough transform; I chose a quite large max_line_gap value in order to connect similar longer lines. I decided to obtain a lower noise in the slope of the lines
* In the draw_lines() method : 
	* I eliminate any lines for which the absolute value of the slope is outside the [25, 40] degrees range 
	* I separate the lines into two lists: negative, and positive slope lines (left and right lanes, respectively)
	* For each of the two lists (negative and positive slopes), I apply a linear regression on all the points to obtain a slope and and intercept (a and b parameters of the y = ax + b equation) for the left and right lanes
	* I add these slopes to the ones obtained in the previous n_frames (currently set to 10), and I choose the mean slope and the mean intercept; these values are used to draw the left and right line on the current frame. This is similar to a moving average of slopes and intercepts, with a window length of n_frames. This results in a smoother lane finding.

An example result of the pipeline applied on the test images is shown below: 

![alt text][pipeline]

---

### 2. Potential shortcomings

The following shortcomings can be identified:
1. Due to the moving average techniques, fast changes in line slopes (during extremely sharp turns) will have a small delay in being detected, directly proportional to the number of frames used for the moving average.
2. If there is a portion of the road where the lanes do not meet the minimum length defined in the Hough transform for more than n_frames, the pipeline will fail to calculate the slope and will eventually crash.

### 3. Possible improvements

Possible improvements are:
1. Adaptive thresholds in the Canny and HSL operations could eliminate the need to use the moving average technique, and produce robust estimations of the slopes on each frame.
2. I could check if lanes are correctly identified on each frame, and if not, draw lanes from the last previous correctly calculated frames.
3. In the linear regression step, I could add weights according to line lenghts, such that very short lines will not affect the slope calculation as much.
