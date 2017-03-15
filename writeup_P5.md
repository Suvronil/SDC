
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


---
###Writeup / README


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I finally extracted the following features and concatenated them to for the final feature vector.
1. HOG feature based on Grayscale of the image.
2. HOG feature based on the S-channel, after converting the image to HLS colorspace.
3. HOG feature based on the Y-channel, after converting the image to YCr_Cb colorspace.
4. Color histogram and spatial binning of the image


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally decided to go ahead with the following combination.

`orientations` : 8
`pixels_per_cell` : 8
`cells_per_block` : 2

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I collected the data in with the above mentioned features and normalized them using `normalize_feature` function .I trained a neural network with 1024 neurons and dropout layer with adam optimizer, using Keras. I maintained following values for these parameters.

keep_prob : Input units to drop 0.2 or 20%.
Loss metric = binary cross-entropy
Learning rate of the adam optimizer = 0.0001
epoch = 15 (Although 5 would have sufficed, I feel)




###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After experimenting for sometime, I decided

1. to search only on the lower half of the image as this portion consists of the car and the road.
2. I divided the lower half of the image into 3 sections or ranges of Y values and decided to use 3 different window sizes((50,50*1.3),(75,75*1.3),(100,100*1.3)), one in each section, but with different overlapping ratio.*
3. Smaller window sizes are used near the middle section of the image and as we go further down the image, the window size increases. This helps us to detect vehicles when it's near to the camera and also the ones, far way and appears smaller.

The code for this can be found in the function test_image.

I decided to go with 0.9 overlapping, as this helps us to detect the edges of the vehicle more precisely, though it costs us more computing time and makes the overall detection process slower.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of neural network classifier, I decided to add dropout layer, so that it doesn't overfit the training images and trained it with adam optimizer with learning rate of 0.0001.

Following is how the pipeline works.
1. `test_image` function receives the image  and passes it to the slide_window function to obtain the list of windows.
2. The portion of the image in these windows are then sent to window_test function for classification.
3. If it's classified as vehicle, it's added to the list bounding boxes.
4. Once all the windows are classified and bounding boxes are found, a heatmap is created. 
5. This heatmap is initially passed to scipy.ndimage.measurements.label()`to generate labels or clusters.

6. I then used a function 'apply_threshold' to generate the threshold for the image, dynamically. This was done so, as for a video, using a static threshold, across all the frames, was not giving a good result and too many false postives were appearing on the images. 'apply_threshold' function gives an optimized threshold based on the detection, achieved for that particular image, so that only the pixels that represents the cars near the camera gets detected.

I measured heat per pixel in the clusters found after applying label function on the heatmap and then applied the follwing condition, to compare heat per pixel in that cluster and avg heat across all clusters.

        if labels == 1:                    # To counter a scenario, where no car is present, yet a false positive                                                         occurs
        
             if((3 > heat_per_pixel_in_cluster)):
                list_heat_per_pixel_in_cluster.append(clusterno)
                labelmap[labelmap==clusterno] = 0
        
        
                                                # For scenarios, where multiple false positives occurs

        elif (labels>1 and avg_heat_per_pixel<2.8): 
            if((1.5 > heat_per_pixel_in_cluster) or (heat_per_pixel_in_cluster < avg_heat_per_pixel*0.7)):
                list_heat_per_pixel_in_cluster.append(clusterno)
                labelmap[labelmap==clusterno] = 0
               
                                                 # For scenarios where both true positives and false positives occur      
               
        else: 
            if ((4 > heat_per_pixel_in_cluster)):
                if(heat_per_pixel_in_cluster < avg_heat_per_pixel*0.7):
                    list_heat_per_pixel_in_cluster.append(clusterno)
                    labelmap[labelmap==clusterno] = 0

From my experience of testing, this on video, I found that false positives usually appear with 1-1.5 value of heat per pixel and can be safely removed. In case, any false positive like, distant cars or cars, being driven on the road beside, gets detected with higher heat per pixel value, it'll be possible to exclude it by the second condition (heat_per_pixel_in_cluster < avg_heat_per_pixel*0.75).


For example, below is what I get, when I test this on test5.jpg. It detects 3 clusters (2 cars and 1 false positive). Yet, based on this thresholding condition, it removes cluster 3, marked under cluster having below avg heat per pixel. 

deep learningwith window size50 on test5.jpg
Heat per pixel count for cluster 1:  5.75120399037
Heat per pixel count for cluster 2:  6.09472049689
Heat per pixel count for cluster 3:  2.34657039711
avg. heat per pixel : 5.55319344534
No. of detected clusters :  3
clusters having below avg heat per pixel : [3]

Following is an example of clusters detected before thresholding (taken from heatmap) and clusters remaining after thresholding and removal of false positive. Image used is the test image test3.jpg.


![test 3 example](https://cloud.githubusercontent.com/assets/26251015/23975427/a2d82aca-0a06-11e7-903a-1cd73c0bb6e6.png)

6. This threshold is then used on the heatmap to nullify all the other values.

labelmap[labelmap==clusterno] = 0 Labelmap is the image like array received from labels() function.

After this the labelmap is passed again to labels() function to generate the final_labels, which has the final clusters with cars.

7. This labels are then used to draw the bounding boxes on the image using the draw_labeled_bboxes function.


`
Final image output example : 


![deep learning with window size60 on test1](https://cloud.githubusercontent.com/assets/26251015/23975474/f844d8be-0a06-11e7-9ee7-8990dd5bbddc.jpg)


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)


Here's a [link to my video result]https://github.com/Suvronil/SDC/blob/master/project_video_output.mp4(SDC/project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map , using the dynamically generated threshold from apply_threshold function , to identify vehicle positions.  

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Finding optimized threshold was the major obstacle , while I was working on the video. A single threshold was showing minimal or no false positive in one section, but too many false positives in a different part of the video. It became more problematic, as the car passes through different road conditions and the vehicles drive by, from the opposite direction on the left hand side. 

Using dynamic thresholding to a good extent took care of this problem.


My pipeline will likely fail to detect any vehicle that passes horizontally. So if a vehicle comes suddenly passing horizontally, near the turn at road, classifier will possibly fail, as it was trained with images of the vehicles, taken from behind.

One way to make this pipeline more robust would be to make the sliding window search better and dynamic, so that it makes better choices to decide upon the window sizes by itself. This will eventually help to optimize the search and will help to reduce  the computing time and also it will be effective in other driving scenarios, apart from highways.For example, if we are driving in city, we do not have the scope of ignoring the upper half of the image always.

