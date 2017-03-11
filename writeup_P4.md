
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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd cell of the jupyter notebook.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.



###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Following is an example where two test images were corrected and the distotion corrected images show the change of positions of objects near the corners, which indicates in turn that the initial image had radial distrtion.

![undistorted example](https://cloud.githubusercontent.com/assets/26251015/23821623/7ae12964-065e-11e7-8a16-12155042b09c.png)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image . I defined 4 function in cell 5 of P4.ipynb to apply these threshold. Outputs of these 4 functions were combined in the following statement to generate the final binary image.

combined[(((gradx == 1) & (grady == 1))|((color_binary == 1)&(mag_binary == 1)))] = 1  , 

here,
gradx,grady,color_binary,mag_binary are the outputs of the 4 thresholding functions.

gradx = Thresholded binary image, received after taking fradient on x direction by applying sobel operator.

grady = Thresholded binary image, received after taking fradient on y direction by applying sobel operator.

color_binary = Thresholded binary image, where threshold has been applied on the R channel of BGR image and S channel of HLS                image, after converting the BGR image. 

mag_binary = Thrsholded binary image, where thresholds have been applied over magnitude of the gradient.


Here's an example of my output for this step. The below images show that line markers have been identified under different conditions , successfully.

![thresholded](https://cloud.githubusercontent.com/assets/26251015/23822285/22d14d14-0670-11e7-96cf-2937bc441516.png)

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform`, which appears in lines 4 through 8 in 5th code cell of the IPython notebook).  The `perspective_transform` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

Here, imshape is the shape of the image, consisting of row(imshape[0]) and colomn(imshape[1]).

```
src = np.float32([[(imshape[1]/2)-70,(imshape[0]-50)/1.5],[(imshape[1]/2)+110,(imshape[0]-50)/1.5],
                       [imshape[1]/2-500,(imshape[0]-50)],
                       [(imshape[1]/2+600),(imshape[0]-50)]])
    
dst = np.float32([[0,0],[1280,0],[0,720],[1280,720]])


```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![transformed image](https://cloud.githubusercontent.com/assets/26251015/23822638/8d18942e-0676-11e7-9d8b-482356c95930.png)



####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used `find_lane` function to fit my lane lines with a 2nd order polynomial in the following manner.

1. First it takes histogram of the binary image to find the concentration of the lane pixels along te axis.
2. Then it splits the histogram in two halfs, so that left and right lanes can be identified separately.
3. It identifies the axis along which, the right and left half of the histogram have maximum pixels.
4. Then it takes a sliding window approach to find the lane pixels.
5. After this, np.polyfit() function is used to fit these pixels to a second degree polyomial.



####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 149 through 178 in cell 5, within function final_plot.

1. A binary image was created.
2. For y axis co-ordinates, X co-ordinates were derived using second order polynomial.
3. Scaling factors were calculated.
4. These pixel space co-ordinates were multiplied with scaling facors to convert into real world values.
5. Using these real world values, radius of curvature was calculated.

 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally plotting of the marked lane on the original image is done using the function final_plot. Following is a sample image.

![vlcsnap-2017-03-11-17h48m54s904](https://cloud.githubusercontent.com/assets/26251015/23823186/0f42d3ae-0683-11e7-927a-1af06797a405.png)

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Video has been submitted with this submission.

Here's a [link to my video result](https://github.com/Suvronil/SDC/blob/master/project_output.mp4)(SDC/project_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The same pipeline, I tried over the challenge video and harder challenge video and the results were least to say catastrophic. This shows that this pipeline fails to detect the lines where line is very curvy or has other lines parallal to the lane markings.

I feel , a way to deal with the difficulties, shown by challenge video would be to make color thresholding process better.  

For harder challenge, a more sophisticated process is required to find parallal lines. 



