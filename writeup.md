## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/bar_training_data.png "Training Data Visualizaton"
[image2]: ./writeup_images/bar_validation_data.png "Validation Data Visualizaton"
[image3]: ./writeup_images/bar_test_data.png "Test Data Visualizaton"
[image4]: ./Sign_images/speed20.png "Traffic Sign 1"
[image5]: ./Sign_images/speed30.png "Traffic Sign 2"
[image6]: ./Sign_images/speed50.png "Traffic Sign 3"
[image7]: ./Sign_images/speed60.png "Traffic Sign 4"
[image8]: ./Sign_images/speed70.png "Traffic Sign 5"
[image9]: ./writeup_images/im1_softmax.png "Sign 1 Softmax Probabilities"
[image10]: ./writeup_images/im2_softmax.png "Sign 2 Softmax Probabilities"
[image11]: ./writeup_images/im3_softmax.png "Sign 3 Softmax Probabilities"
[image12]: ./writeup_images/im4_softmax.png "Sign 4 Softmax Probabilities"
[image13]: ./writeup_images/im5_softmax.png "Sign 5 Softmax Probabilities"

## Rubric Points

The grading rubric can be seen [here](https://review.udacity.com/#!/rubrics/481/view). Each point will be considered in this writeup.

---

## Data Set Summary & Exploration

1. Provide a basic summary of the data set.

The python 'len' and 'max' functions along with the 'shape' method were used to calulate the following statistics regarding the data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32 x 32.
* The number of unique classes/labels in the data set is 43.

2. Include an exploratory visualization of the dataset.

A bar chart for each data set is included below. The charts show that training data and test data are very similar in proportion. The validation set is a bit smaller (~10% of the training set, ~33% of the test set size) and the bar chart shows that it has less discretization and a coarser resolution. This leads to the proportions being a bit different than the other data sets. 

![Training Data][image1]
![Validation Data][image2]
![Test Data][image3]

Design and Test a Model Architecture

1. Preprocessing

The selected preprocessing was very straightforward. Only normalization was ultimately used on the data. 

Early efforts to consider preprocessing seemed to make no difference in performance. Likely this was a function of a poorly built network at the time. After building and tuning the network, the accuracy targets had already been exceeded, and no further work was done on preprocessing. This is an area that could be mined for potential future performance improvements. 

2. Model Architecture

The model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation			| tanh											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| Activation			| tanh											|
| Pooling				| 2x2 stride, outputs 5x5x16					|
| Flatten				| output = 400									|
| Fully Connected 		| output = 120									|
| Activation 		 	| tanh											|
| Fully connected		| output = 84									|
| Activation 		 	| tanh											|
| Fully connected		| output = 43 									|


3. Training the Model 

To train the model, a series of tests were performed to select the learning rate, batch size, and activation layers. Early results showed that batch size was much less of a discriminator than the learning rate and activation layers, so the default 128 batch size was selected. Establishing the order of magnitude of the training rate (~0.001) was fairly straightforward. The activation layer took significant trial and error, as there are a great many possible cominbations. It turned out that making them all the 'tanh' showed a lot of promise. It was then possible to refine the learning rate to 0.0008 for slightly better performance.

4. Describe the Approach  

A paper by [Sermanet and LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) described an approach using the 'tanh' function for this class of sign recongition problem. This appeared to work well based on training. 

Ultimately 30 epochs were used. Most of the accuracy was achieved in 10-11 epochs, but a slight bump upwards did appear around epoch 28 (from about 0.950 to 0.955 on the validatation set).  

Final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.955
* test set accuracy of 0.934

With regards to the evoluation of the selection of the activation layers, the following is a description of the process:
Initally all four activation layers were chosen be dropout functions. A large number of values were tested for the keep probability but none seemed to provide a good performance (validation accuracy above 0.9).

Making all four layers 'relu' did not lead to good performance either. 

Selecting 'tanh' for the activation layers was much more successful than any other approach attempted. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


