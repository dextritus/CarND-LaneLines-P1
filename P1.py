
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# get_ipython().magic('matplotlib inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def line_slopes(lines):
    """
    returns a list of slopes for a given list of lines
    """
    slopes = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slopes.append(np.degrees(np.arctan(float((y2-y1))/(x2-x1))))

    return slopes

def line_slope(line):
    """
    returns a list of slopes for a given list of lines
    """
    for x1, y1, x2, y2 in line:
        return np.degrees(np.arctan(float((y2-y1))/(x2-x1)))

avg_a_left = []
avg_b_left = []

avg_a_right = []
avg_b_right = []



def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
    neg_slope  = []
    pos_slope  = []

    img_x = img.shape[1]
    img_y = img.shape[0]

    #remove lines that are close to horizontal 
    no_horiz = [line for line in lines if abs(line_slope(line)) > 25 and abs(line_slope(line)) < 40]
    for line in no_horiz:
        if line_slope(line) < 0: neg_slope.append(line)
        else: pos_slope.append(line)

    avg_neg_slope = 0.0
    for line in neg_slope:
        avg_neg_slope += line_slope(line) / len(neg_slope)

    avg_pos_slope = 0.0
    for line in pos_slope:
        avg_pos_slope += line_slope(line) / len(pos_slope)

    x_v = []
    y_v = []
    for x in neg_slope:
        x_v.append(x[0][0])
        x_v.append(x[0][2])
        y_v.append(x[0][1])
        y_v.append(x[0][3])

    neg_coeff = np.polyfit(x_v, y_v, 1)
    avg_a_left.append(neg_coeff[0])
    avg_b_left.append(neg_coeff[1])

    if (len(avg_a_left) > 10): avg_a_left.pop(0)
    if (len(avg_b_left) > 10): avg_b_left.pop(0)
    
    cur_a_left = np.median(avg_a_left)
    cur_b_left = np.median(avg_b_left)

    # cur_a_left = neg_coeff[0]
    # cur_b_left = neg_coeff[1]

    x_v = []
    y_v = []

    for x in pos_slope:
        x_v.append(x[0][0])
        x_v.append(x[0][2])
        y_v.append(x[0][1])
        y_v.append(x[0][3])
    
    pos_coeff = np.polyfit(x_v, y_v, 1)
    avg_a_right.append(pos_coeff[0])
    avg_b_right.append(pos_coeff[1])

    if (len(avg_a_right) > 10): avg_a_right.pop(0)
    if (len(avg_b_right) > 10): avg_b_right.pop(0)
    
    cur_a_right = np.mean(avg_a_right)
    cur_b_right = np.mean(avg_b_right)
    # cur_a_right = pos_coeff[0]
    # cur_b_right = pos_coeff[1]

    # neg_coeff = np.polyfit([neg_bottom_left_x, neg_top_right_x], [neg_bottom_left_y, neg_top_right_y], 1)
    # pos_coeff = np.polyfit([pos_bottom_right_x, pos_top_left_x], [pos_bottom_right_y, pos_top_left_y], 1)

    # print(neg_coeff[0])

    # cv2.line(img, (neg_bottom_left_x, neg_bottom_left_y), (neg_top_right_x, neg_top_right_y), [0, 255, 0], 10)
    
    cv2.line(img, (int((0.6*img_y - cur_b_left)/cur_a_left), int(0.6*img_y)), \
                  (int((img_y - cur_b_left)/cur_a_left), img_y), color, thickness)

    cv2.line(img, (int((0.6*img_y - cur_b_right)/cur_a_right), int(0.6*img_y)), \
                  (int((img_y - cur_b_right)/cur_a_right), img_y), color, thickness)

    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.9, β=1.0, λ=0.0):
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[32]:



def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    ## convert the image to grayscale, in preparation for Canny
    gray = grayscale(image)
    
    gray_hist = np.zeros_like(gray);
    cv2.equalizeHist(gray, gray_hist);
    
    ## canny filter for edge detections
    #  first, gaussian blur
    gaussian_kernel = 11
    blur_gray = gaussian_blur(gray_hist, gaussian_kernel)
    
    #  canny filter
    canny_low = 100
    canny_high = 200
    edges = canny(blur_gray, high_threshold=canny_high, low_threshold=canny_low)
    
#     plt.figure(figsize = (10, 10))
#     plt.imshow(edges, origin = 'upper')
    # crop the image for the region of interest
    im_x = image.shape[1]
    im_y = image.shape[0]
    
#     print("image width ", im_x)
#     print("image height ", im_y)
    
    crop_verts = np.array([[(100.0, im_y), (im_x-100, im_y), (600, 350), (375, 350), (100.0, im_y)]], dtype = np.int32)
    cropped = np.zeros_like(edges)
    cropped = region_of_interest(edges, crop_verts)
#     plt.imshow(cropped)
    
    ## recover the lines from the Hough space
    max_line_gap = 15
    min_line_len = 20
    rho = 1
    theta = 1*np.pi/180
    threshold = 10
    
    houghed_img = hough_lines(cropped, max_line_gap=max_line_gap, min_line_len = min_line_len, rho = rho, theta = theta, threshold = threshold)
    
#     plt.figure(figsize = (10, 10))
#     plt.imshow(houghed_img, origin = 'upper')
    result = weighted_img(houghed_img, image)
    return result


def process_image_test(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    im_x = image.shape[1]
    im_y = image.shape[0]

    ## convert the image to grayscale, in preparation for Canny
    gray = grayscale(image)
    
    ## canny filter for edge detections
    #  first, gaussian blur

    crop_verts = np.array([[(100.0, im_y), (im_x-50, im_y), (650, 350), (300, 350), (100.0, im_y)]], dtype = np.int32)    

    # gray_hist = np.zeros_like(gray)
    # cv2.equalizeHist(gray_cropped, gray_hist)

    # plt.figure()
    # plt.imshow(gray_cropped)
    
    gaussian_kernel = 11
    blur_gray = gaussian_blur(gray, gaussian_kernel) 
    
    #  canny filter
    canny_low = 80
    canny_high = 189
    edges = canny(blur_gray, high_threshold=canny_high, low_threshold=canny_low)
    
    # plt.figure(figsize = (10, 10))
    # plt.imshow(edges, origin = 'upper')
    # crop the image for the region of interest
    canny_cropped = region_of_interest(edges, crop_verts)

#     print("image width ", im_x)
#     print("image height ", im_y)
    
    # cropped = np.zeros_like(edges)
    # cropped = region_of_interest(edges, crop_verts)
#     plt.imshow(cropped)
    
    ## recover the lines from the Hough space
    max_line_gap = 250
    min_line_len = 20
    rho = 1
    theta = 1*np.pi/180
    threshold = 20
    

    houghed_img = hough_lines(canny_cropped,max_line_gap=max_line_gap, min_line_len = min_line_len, rho = rho, theta = theta, threshold = threshold)
    # plt.figure(figsize = (10, 10))
    # plt.imshow(houghed_img, origin = 'upper')
    result = weighted_img(houghed_img, image)

    # plt.figure(figsize = (10, 10))
    # plt.imshow(result, origin = 'upper')
    # print(text)
    return result

img = mpimg.imread('test_images/solidYellowCurve.jpg')

def rgb2hls(image):

    im_y = image.shape[0]
    im_x = image.shape[1]

    img2 = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    yellow = np.uint8([[[255, 255, 0]]])
    yellow_conv = cv2.cvtColor(yellow, cv2.COLOR_RGB2HLS)

    white = np.uint8([[[255, 255, 255]]])
    white_conv = cv2.cvtColor(white, cv2.COLOR_RGB2HLS)
    low_w = np.uint8([0, 210, 0])
    high_w = np.uint8([200, 255, 255])
    # print(white_conv)
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.imshow(img2, origin = 'upper')
    low_y = np.uint8([10, 0, 100]);
    high_y = np.uint8([50, 255, 255]);
    yellow_mask = cv2.inRange(img2, low_y, high_y)
    white_mask = cv2.inRange(img2, low_w, high_w)
    total_mask = cv2.bitwise_or(white_mask, yellow_mask)
    final_img = cv2.bitwise_and(image, image, mask = total_mask)

    gray = grayscale(final_img)
    blur_gray = gaussian_blur(gray, 11) 
    edges = canny(blur_gray, 80, 160)
    crop_verts = np.array([[(0.1*im_x, im_y), (int(0.95*im_x), im_y), (int(0.55*im_x), int(0.6*im_y)), (int(0.4*im_x), int(0.6*im_y)), (int(0.1*im_x), im_y)]], dtype = np.int32)    

    canny_cropped = region_of_interest(edges, crop_verts)
    # edges = canny_cropped
    ########Hough
    max_line_gap = 250
    min_line_len = 20
    rho = 1
    theta = 1*np.pi/180
    threshold = 10

    houghed_img = hough_lines(canny_cropped,max_line_gap=max_line_gap, min_line_len = min_line_len, rho = rho, theta = theta, threshold = threshold)
    result = weighted_img(houghed_img, image)

    # plt.subplot(2, 1, 2)
    # plt.imshow(final_img, origin = 'upper')
    # print(yellow_conv)
    # print(yellow_conv[0])
    # return weighted_img(final_img, image)
    return result
    # return np.dstack((edges, edges, edges))

img = mpimg.imread('test_images/solidYellowCurve2.jpg')
# image_test = process_image_test(img)

white_output = 'test_videos_output/test.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(rgb2hls) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(rgb2hls)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(rgb2hls)
challenge_clip.write_videofile(challenge_output, audio=False)



rgb2hls(img)

plt.show()

