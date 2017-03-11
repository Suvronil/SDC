
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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

