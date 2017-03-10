#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### P2.ipynb notebook consists of the code for this project.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the code cell ?? of the jupyter notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (39209,32,32,3)
* The size of test set is (12630,32,32,3)
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cell ??? of the IPython notebook.  

I have used histograms mostly to explore no. of samples for each class in the dataset. It has been most useful to observe
the gap between ovr-sampled classes and under-sampled classes.

I included a small piece of code to check the initial sizes of the images befoe being resized to (32,32) and created a scatter plot. Though it was not of any use at all. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the code cell ??? of the IPython notebook.

Following techniques were used.

1. Data generation - Additional samples have been generated for those classes which has much lesser no. of samples by rotating the images and increasing the brightness on blue channel.

2. Shuffling - Shuffling has been implemented to ensure (a) the model doesn't overfit the data. (b) model gets trained on data from all the classes, as the data in the training set is set in an order. Therefore, it's important to shuffle and then to split it in training and validation set.


3. Grayscaling - Grayscaling has been done inside the model. Grayscaling the image, helps us to deal with lesser data, yet the data we need. It in most of the cases retain informations like shapes and patters and only looses color, which as a feature do not help all the time. This makes the processing faster and also consumes less memory.


4. Normalization - Normalization has been done inside model and it has been implemented by dividing the image matrix values by 255 and then subtracting 0.5, both elementwise. This puts the normalized values between -0.5 and +0.5 with mean zero. This ensures that all the features in the feature vector contributes proportionately in determination of image class. 




####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the  code cell ???? of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using train_test_split method of scikit-learn and 30% of the training data was randomly selected to form the validation set.

My final training set had 44619 number of images. My validation set and test set had 11763 and 12630 number of images.

The code cell# ??? of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data to reduce the gap between no. of samples, between oversampled classes and under-sampled classes. To add more data to the the data set, I warped samples of under-sampled classes by rotating it with random angle and by increasing brightness on blue channel.



The difference between the original data set and the augmented data set is the following ... 

Comparison of the no. of samples present against each class before and after balancing... 
Class Id :  0 		No. of samples before balancing:  136 		No. of samples after balancing :  545
Class Id :  1 		No. of samples before balancing:  1560 		No. of samples after balancing :  1560
Class Id :  2 		No. of samples before balancing:  1595 		No. of samples after balancing :  1595
Class Id :  3 		No. of samples before balancing:  968 		No. of samples after balancing :  968
Class Id :  4 		No. of samples before balancing:  1405 		No. of samples after balancing :  1405
Class Id :  5 		No. of samples before balancing:  1285 		No. of samples after balancing :  1285
Class Id :  6 		No. of samples before balancing:  280 		No. of samples after balancing :  1120
Class Id :  7 		No. of samples before balancing:  1005 		No. of samples after balancing :  1005
Class Id :  8 		No. of samples before balancing:  972 		No. of samples after balancing :  972
Class Id :  9 		No. of samples before balancing:  1007 		No. of samples after balancing :  1007
Class Id :  10 		No. of samples before balancing:  1440 		No. of samples after balancing :  1440
Class Id :  11 		No. of samples before balancing:  919 		No. of samples after balancing :  919
Class Id :  12 		No. of samples before balancing:  1434 		No. of samples after balancing :  1434
Class Id :  13 		No. of samples before balancing:  1484 		No. of samples after balancing :  1484
Class Id :  14 		No. of samples before balancing:  550 		No. of samples after balancing :  550
Class Id :  15 		No. of samples before balancing:  445 		No. of samples after balancing :  1780
Class Id :  16 		No. of samples before balancing:  280 		No. of samples after balancing :  1120
Class Id :  17 		No. of samples before balancing:  792 		No. of samples after balancing :  792
Class Id :  18 		No. of samples before balancing:  822 		No. of samples after balancing :  822
Class Id :  19 		No. of samples before balancing:  145 		No. of samples after balancing :  580
Class Id :  20 		No. of samples before balancing:  254 		No. of samples after balancing :  1016
Class Id :  21 		No. of samples before balancing:  229 		No. of samples after balancing :  916
Class Id :  22 		No. of samples before balancing:  291 		No. of samples after balancing :  1164
Class Id :  23 		No. of samples before balancing:  359 		No. of samples after balancing :  1436
Class Id :  24 		No. of samples before balancing:  195 		No. of samples after balancing :  780
Class Id :  25 		No. of samples before balancing:  1049 		No. of samples after balancing :  1049
Class Id :  26 		No. of samples before balancing:  423 		No. of samples after balancing :  1692
Class Id :  27 		No. of samples before balancing:  168 		No. of samples after balancing :  672
Class Id :  28 		No. of samples before balancing:  371 		No. of samples after balancing :  1484
Class Id :  29 		No. of samples before balancing:  186 		No. of samples after balancing :  744
Class Id :  30 		No. of samples before balancing:  339 		No. of samples after balancing :  1356
Class Id :  31 		No. of samples before balancing:  549 		No. of samples after balancing :  549
Class Id :  32 		No. of samples before balancing:  171 		No. of samples after balancing :  684
Class Id :  33 		No. of samples before balancing:  482 		No. of samples after balancing :  1928
Class Id :  34 		No. of samples before balancing:  291 		No. of samples after balancing :  1164
Class Id :  35 		No. of samples before balancing:  865 		No. of samples after balancing :  865
Class Id :  36 		No. of samples before balancing:  280 		No. of samples after balancing :  1120
Class Id :  37 		No. of samples before balancing:  150 		No. of samples after balancing :  600
Class Id :  38 		No. of samples before balancing:  1447 		No. of samples after balancing :  1447
Class Id :  39 		No. of samples before balancing:  212 		No. of samples after balancing :  848
Class Id :  40 		No. of samples before balancing:  252 		No. of samples after balancing :  1008
Class Id :  41 		No. of samples before balancing:  183 		No. of samples after balancing :  732
Class Id :  42 		No. of samples before balancing:  176 		No. of samples after balancing :  704



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ???? cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 grayscale image   							| 
| Convolution1 5x5     	| 1x1 stride, valid padding, 	|
| RELU
| Max Pool 2x2     	| 1x1 stride, valid padding, 	|
| Max Pool 2x2     	| 1x1 stride, valid padding, 	|
| Convolution2 5x5     	| 1x1 stride, valid padding, 	|
| RELU					|												|
| Max pool 2x2      	| 1x1 stride,   				|
| Flatten 1 = Flatten output of convolution layer 1
| Flatten 2 = Flatten output of convolution layer 2
| Concatenation layer = concatenate flatten 1 and flatten 2
| Fully connected	layer with 1024 neurons
| RELU
| Output layer with 43 neurons
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used following parameters.

Batch size : 1000 Epochs : 15
Optimizer : Adam optimizer with learning rate 0.0001.
Hyperparameters :
mu = 0 sigma = 0.1 for convolutional layers weight initialization.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:
* validation set accuracy of 99.1%.
* test set accuracy of 93.8%.
* external data test set accuracy of 100%.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First model, that I tried was a simple neural network. I tried it to see how a normal neural network performs over this.
Gradually I included a convolutional layer to see how it improves the overall accuracy.

* What were some problems with the initial architecture?

Choice of a good sensible activation function is critical. My initial model suffered a lot on poor accuracy, as i was using sigmoid as activation function and later turned to 'tanh' and finally to RELU.

* Which parameters were tuned? How were they adjusted and why?

Following parameters were tuned usually.

No. of epochs  - To train the model further to lower the loss and increase the accuracy.

Learning rate of Adam optimizer - Learning rate of an optimizer is critical in decreasing the loss gradually, to reach near the global minima of loss.

No. of neurons in fully connected layers - To make the classifier more efficient.



* What are some of the important design choices and why were they chosen? 

Implementing the inception technique as described in Yann Lecun's paper boosted the accuracy to a good extent.
Concatenating the lower level features from convolutional layer 1 with the higher level features from convolutional layer 2 helped the classifier to correctly classify more number of test samples.

I used max pooling to prevent overfitting.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose 5 images similar to the images taken from German traffic sign.

The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.8%.


The code for making predictions on my final model is located in the ??? cell of the Ipython notebook.


