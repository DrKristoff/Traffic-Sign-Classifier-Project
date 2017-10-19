#**Traffic Sign Recognition** 

##Project 2 Writeup

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

[image1]: ./histogram.png "Histogram"
[image4]: ./test_signs/12_priority.jpg "Traffic Sign 1"
[image5]: ./test_signs/14_stop.jpg "Traffic Sign 2"
[image6]: ./test_signs/20_dang_curve.jpg "Traffic Sign 3"
[image7]: ./test_signs/28_children.jpg "Traffic Sign 4"
[image8]: ./test_signs/40_roundabout.jpg "Traffic Sign 5"
[image9]: ./test_signs/01_30speed.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/DrKristoff/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,360
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of example images are included in the dataset for training examples.  

![Sign Histogram][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I created a prepocess function that first converts the image to grayscale and then normalizes the image.  To normalize the image data, I subtract 128 from the rgb value so that rather than spanning 0 to 256, it spans -128 to 128.  I then divided the value by 128 so that it spans from -1 to 1.  This allows the data to have a zero mean and equal variance.  For the final training, I actually commented out the grayscale portion as an experiment and still hit the accuracy target.  Converting to grayscale would improve performance.  


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers, which were based off the Lenet-5 Architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten					|400 outputs												|
| Fully connected		| 120 outputs        									|
| RELU with Dropout					|70% keep probability										|
| Fully connected		| 84 outputs        									|
| RELU with Dropout					|70% keep probability										|
| Fully connected		| 43 outputs        									|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the hyperparameters of 20 epochs, a batch size of 128, and a learning rate of 0.001. Through some simple iterations, I found that any less than 20 epochs wouldn't get the accuracy target I was seeking, and adjusting the learning rate higher or lower caused much slower convergence, or no convergence within a reasonable amount of time.  I used the same optimizer as the MNIST data example, using the Adam Optimizer.  

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 95.4%
* test set accuracy of 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

My whole design started from the LeNet-5 Architecture, at the recommendation of the course documents.  LeNet has proven very capable in the fields of OCR, character recognition and image detection.   As such, it seemed to be a perfect candidate for a traffic sign detection algorithm.  Not only that, but it is often referred to as a great "first CNN," because of its straightforward and small nature.  It is ideal for learning Convolutional Neural Networks.  Convolutional Neural Networks are well suited for image recognition type analyses due to their ability to look at groups of neighboring pixels to formulate small features that contribute to the overall picture.  

After setting up all the code to work with the new dataset, I ran the test with the default values of epochs = 10, batch size = 128, learning rate = 0.001.  The accuracy rate was around 90% so I began to play around with the learning rate and found that the default value seemed to converge the fastest.  After doing some research online, I decided to change the last two ReLU layers to include dropout between the fully connected layers.  I started with a keep rate of 90% and this slightly improved my accuracy.  I then dropped the keep rate to 70% which kicked my accuracy up to over 93%, peaking at 96% if I ran it out to 30 epochs.  This quick result seemed to validate the fact that the LeNet CNN was a good algorithm choice for this problem.  

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]


These were all screenshots from a large chart of german signs.  Looking back, this probably made the signs very favorable for classifying, since they had white backgrounds and their alignment and lighting was ideal.  Near the end of this project, I made this realization and replaced one of the images with the roundabout image which was an actual picture taken of a sign.  This one would presumably be the most difficult to detect, since the other 4 were absolutely ideal.  I'll discuss this more down with the softmax section.  

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Sign      		| Priority sign   									| 
| Stop sign     			| Stop sign 										|
| Right Turn					| Right Turn											|
| Children crossing	      		| Children crossing					 				|
| Mandatory Roundabout			| Mandatory Roundabout      							|
| 30 Speed Limit			| 30 Speed Limit      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%, which aligns similarly with the test and validation sets.  

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

~~For each of the softmax probabilities below, there was some confusion on my part on the results delivered by the algorithm.  The distribution and values didn't make sense to what I would intuitively expect.  The results were low percentage probabilities for images that were very ideal.  Some values returned negative (N/A below).  The algorithm still was successful in determing the most likely sign, but I need to do further research on how to interpret (or correct) the values shown below.~~

After fixing my code I was able to correctly calculate the Softmax probabilities.  My initial results with 5 examples each yielded 100% accuracy, which is great, but I wanted to make sure that it was actually working.  As I mentioned previously, because I picked really favorable pictures for the test, the algorithm wasn't really pushed to the limit.  Because of this, I searched for a 6th picture that was at skewed and at an angle with more distracting background.  I expected that the confidence on this picture would be much less, which proved true.  This picture was the third image below.  

For the first image, the model predicts with 99.9% confidence that it's a Priority sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority   									| 
| .00     				| Traffic Signals 										|
| .00					| Yield											|
| .00	      			| No Passing					 				|
| .00				    | End of No Passing      							|


For the second image, the model predicts with 99.9% confidence that it's a Stop Sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop Sign   									| 
| .00     				| 29 										|
| .00					| 25											|
| .00	      			| 3					 				|
| .00				    | 1      							|

For the third image, the model predicts with 48% confidence that it's a Yield sign.  This lower probability was expected because I purposely chose a more difficult image to test how well the algorithm worked.  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .48         			| Yield   									| 
| .38     				| No Passing 										|
| .14					| Turn Left											|
| .00	      			| Keep Right					 				|
| .00				    | Keep Left      							|

For the fourth image, the model predicts with 99.9% confidence that it's a Dangerous Curve Right sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Dangerous Curve Right   									| 
| .00     				| Children Crossing 										|
| .00					| Slippery Road											|
| .00	      			| End of No Passing					 				|
| .00				    | Bicycle Crossing      							|

For the fifth image, the model predicts with 99.9% confidence that it's a Children Crossing sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Children Crossing   									| 
| .00     				| Dangerous Curve Right 										|
| .00					| Bicycle Crossing											|
| .00	      			| Pedestrians					 				|
| .00				    | Slippery Road      							|

For the sixth image, the model predicts with 99.9% confidence that it's a Mandatory Roundabout sign

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Mandatory Roundabout   									| 
| .00     				| Keep Right 										|
| .00					| Keep Left											|
| .00	      			| Speed 50					 				|
| .00				    | Road Narrows Right      							|


