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

The code for this step is contained in the code cell 2-4 of the jupyter notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is (39209,32,32,3)
* The size of test set is (12630,32,32,3)
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cells 3,4,7,10,12 of the IPython notebook.  

I have used histograms mostly to explore no. of samples for each class in the dataset. It has been most useful to observe
the gap between ovr-sampled classes and under-sampled classes.

I included a small piece of code to check the initial sizes of the images befoe being resized to (32,32) and created a scatter plot. Though it was not of any use at all. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the code cell 6 and 13 of the IPython notebook.

Following techniques were used.

1. Data generation - Additional samples have been generated for those classes which has much lesser no. of samples by rotating the images and increasing the brightness on blue channel.

2. Shuffling - Shuffling has been implemented to ensure (a) the model doesn't overfit the data. (b) model gets trained on data from all the classes, as the data in the training set is set in an order. Therefore, it's important to shuffle and then to split it in training and validation set.


3. Grayscaling - Grayscaling has been done inside the model. Grayscaling the image, helps us to deal with lesser data, yet the data we need. It in most of the cases retain informations like shapes and patters and only looses color, which as a feature do not help all the time. This makes the processing faster and also consumes less memory.


4. Normalization - Normalization has been done inside model and it has been implemented by dividing the image matrix values by 255 and then subtracting 0.5, both elementwise. This puts the normalized values between -0.5 and +0.5 with mean zero. This ensures that all the features in the feature vector contributes proportionately in determination of image class. 




####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the  code cell 6 of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using train_test_split method of scikit-learn and 30% of the training data was randomly selected to form the validation set.

My final training set had 44619 number of images. My validation set and test set had 11763 and 12630 number of images.

The code cell# ??? of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data to reduce the gap between no. of samples, between oversampled classes and under-sampled classes. To add more data to the the data set, I warped samples of under-sampled classes by rotating it with random angle and by increasing brightness on blue channel.



The difference between the original data set and the augmented data set is the following ... 

Comparison of the no. of samples present against each class before and after balancing is given in the cell # of jupyter notebook. 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 13 cell of the ipython notebook. 

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

####1. Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult?

I chose 5 images from web, similar to the images taken from German traffic sign. Following are the images.

Image 1 : Class id : 35  # Ahead only

![1](https://cloud.githubusercontent.com/assets/26251015/23820108/534ab57c-0637-11e7-9b38-9ad0f898943b.jpg)

Image 2 : Class id : 9  # No passing

![2](https://cloud.githubusercontent.com/assets/26251015/23820125/bfa7014e-0637-11e7-9ef3-99f90c68dc2f.jpg)

Image 3 : Class id : 17  # No entry

![3](https://cloud.githubusercontent.com/assets/26251015/23820126/bfb4e80e-0637-11e7-879d-d27dbdb997c8.jpg)

Image 4 : Class id : 13  # Turn left ahead

![4](https://cloud.githubusercontent.com/assets/26251015/23820127/bfbd5ffc-0637-11e7-84e7-4a73d24e7021.jpg)

Image 5 : Class id : 14  # Stop

![5](https://cloud.githubusercontent.com/assets/26251015/23820128/bfc016d4-0637-11e7-8f8c-12e15b2d619e.jpg)


There are few things that might make classification difficult. Such as,

1. Too much of background, which we may say in this case will become noise in the data and will make classification difficult.

2. Poor resolution may affect the classification in few cases. Like, while classifying speed limit of 30 km/h and 80km/h,it may fail, due to similarity of shape among 3 and 8. Poor resolution will add up to the chances of mis-classification.

3. Change of perspective also might make classification difficult. Although we can recover this, by training the data with augmented images, where perspecting transform is done.

4. Partial obervation: where possible main content like an arrow of the sign probably has been captured, yet it has missed to capture a minor part of it, may affect the result.If the sign is off-centered in the image for simiar reasons, will also lead to misclassification sometimes. 







####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| No passing     			| No passing 										|
| No entry					| No enty											|
| Turn left ahead	      		| Turn left ahead					 				|
| Stop Sign			| Stop Sign      							|

The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Following are probablities, model showed for the correct class.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Ahead only   									| 
| .95     				| No pasing 										|
| 1					| No entry											|
| 1	      			| Turn left ahead					 				|
| 1				    | Stop sign      							|


