# **Traffic Sign Recognition** 

## My Writeup for Submission

#### Hereby I describe how I finished the 3rd project of this Nanodegree program and how I achieved all the given goals. Honestly speaking, I have run the code both in work space and on my local machine and I have thus two IPython notebook files. Here are the correct links to the [project code](./Traffic_Sign_Classifier_submitted.ipynb) and to [html file](./Traffic_Sign_Classifier.html).

---

**Created on 21.06.2020**
First writeup finished.

[//]: # (Image References)

[image1]: ./writeup_figure/bar_chart_all.png "Visualization"
[image2]: ./writeup_figure/err_rate_curve.png "Error rate"
[image3]: ./raw_img/NoPassing_28x28.jpg "No Passing Sign"
[image4]: ./raw_img/Roadwork_28x28.jpg "Roadwork Sign"
[image5]: ./raw_img/Speed50_28x28.jpg "Speed Limit 50km/h Sign"
[image6]: ./raw_img/Speed100_28x28.jpg "Speed Limit 100km/h Sign"
[image7]: ./raw_img/Stop_28x28.jpg "Stop Sign"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

### Data Set Summary & Exploration

#### 1. By using numpy, I managed to have a basic summary of the traffic sign dataset, after loading it from pickle file.

I used the numpy library to check the numbers of train, validation and test samples, due that the dataset had been split by the provider:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630

I checked if the images had been padded or not, how many classes were there in the entire dataset.

* The shape of a traffic sign image is (32, 32) as RGB image
* The number of unique classes/labels in the data set is 43

#### 2. I also included an exploratory visualization of the dataset.

![alt text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of all the samples in the dataset over all the classes. For that, I concatenated train, validation and test subsets by writing:

```python
y_all = np.concatenate((y_train, y_valid, y_test), axis = None)
plt.figure(1)
plt.hist(y_all, bins=n_classes-1)
```

A randomly chosen image was also displayed in the IPython notebook from one subset along with its class label.

### Design and Test a Model Architecture

#### 1. I chose to pre-process the data only with normalization.

For pre-processing the images, normalization was my only choice for the implementation this time.

To avoid a rollover error due to **uint8** data type, type converting is a must before normalization

```python
X_train_norm, X_valid_norm, X_test_norm = [(X_subset.astype(float) - 128)/ 128 for X_subset in [X_train, X_valid, X_test]]
```

In fact, I have known a lot of data augmentation methods. I think I wouldn't convert the color to grayscale or shift the color too much, because color is one of the most critival factors for traffic sign recognition. A flip in horizontal or vertical direction is also not recommended, because the classifier would misunderstand the information of the sign.

For improvement, a little bit rotation, perspetive transformation and color jittering, I think, is considerable, since traffic signs always show up with such variation in reality.

Making some artificial noise or scaling the images might take effect. But the size input images are so small that the benefits would not be very obvious.


#### 2. I almost kept the whole original architecture, which was used in the LeNet-Lab. Yet I added a dropout layer before the classifier.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, no padding, outputs 28x28x6 	    |
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, no padding, outputs 10x10x16 		|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 					|
| Fully connected		| outputs 120									|
| Fully connected		| outputs 84									|
| Dropout				| keep probability = 0.5 when training			|
| Softmax				| 												|
 
The added dropout managed to learn more redundant features when training and during tht test, the classifier is able to get a better feature representing the images.

#### 3. The training pipeline was set up as same as the one utilized in the LeNet-Lab. 

To train the model, I converted the original label to a one-hot label. The keep probability for dropout was set to 0.5. A cross entropy was calculated over the softmax results passed through the net. To minimize this loss function, I chose Adam-Optimizer (I heard that Adam is smarter than SGD, not sure of the principle) for back propagation.

For calculating the model accuracy, an evaluation function was also created, in which the keep probability for dropout was set to 1.

#### 4. I indeed trained the model iteratively, but fortunately not too many times. The dropout works quite well for me!! 

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.952 
* test set accuracy of 0.933

For each training, I noted the parameters and the validation accuracy as below so that I haven't been lost then.

| Model        			|     Epochs / Batch size / Learning rate 	| Validation Accuracy	|
|:---------------------:|:-----------------------------------------:|:---------------------:|
| LeNet (original)		|    10 / 128 / 0.001 						|   0.891				|
| LeNet (original)    	|    30 / 64 / 0.0005 						|   0.929				|
| LeNet with dropout	|    30 / 64 / 0.0005 						|   0.923				|
| LeNet with dropout	|    50 / 64 / 0.0004 						|   0.952				|

I created a error rate with respect to epochs curve to have an overview of each training. That told me how to adjust the parameters next. It would be better to supervise the procedure online.

![alt text][image2]

Some answers to the questions about the iterative training:
* The first architecture was just the one used in the LeNet-Lab, which was also introduced in the last chapter.
* As for me, the initial architecture was simple and easy to implement. It was actually difficult to push the accuracy higher, now that we could have a performance over 89%.
* In fact, I didn't check the accuracy over train samples until writing this. I don't think a overfitting happend to the my beginning, because the validation accuracy was high enough. I decided to add this dropout because the last feature vectors before classifier have 84 dimensions, which, I think, are too many for an image like traffic sign. More channels could be used to learn more redundant features to improve the recognition rate.
* First I tuned the batch size to 64. This param has generally less influence on result if the batch size is not too small. Then I analysed the validation accuracies over the last 10 epochs. If there is almost no fluctuaion, that means the training is not complete and I will add more epochs. If the validation accuracy fluctuates quite early, or looks saturated, that means the learning rate is too big at that moment and it should decrease accordingly. If the validation accuracy increases relatively slowly in the first several epochs, the learning rate should increase again.
* For my architecture, the most important design is the dropout. During training, half of features are dropped before classifier, which makes the net model learn a better group of weights and bias. And during test, the learned redundant features maximize the power of the net and help 'confirm' the class with higher confidence.


### Test a Model on New Images

#### 1. I found 5 images which were shot in the real German streets. 

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image3]

None of the traffic signs in those images are difficult to classify because they are clear, under good light condition and in a good perspective. Therefore I cropped them and scales them to the size of 28x28

#### 2. Surprisingly for me, my model's predictions on those new images are all correct and the accuracy is 100%

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work     		| Road Work   									| 
| Speed Limit 50		| Speed Limit 50								|
| Speed Limit 100		| Speed Limit 100								|
| Stop sign	      		| Stop sign						 				|
| No Passing			| No Passing      								|


#### 3. The top 5 softmax probabilities for each image along with the sign type of each probability are given below

Because the confidence of my model's first predicted candidate is so high (over 0.9999 for each test images), showing the numbers of other 4 candidates might look exaggerating. Just the list of candidates will be given here


| Label        			|     Predicted Candidates (Descending order by confidence)									| 
|:---------------------:|:-----------------------------------------------------------------------------------------:| 
| Road Work    			| Road Work, Wild animals crossing, Bicycles crossing, Bumpy road, Beware of ice/snow		| 
| Speed Limit 50		| Speed Limit 50, Speed Limit 30, Speed Limit 80, Speed Limit 60, Wild animals crossing		|
| Speed Limit 100		| Speed Limit 100, Speed Limit 80, Speed Limit 50, Speed Limit 120, Speed Limit 30			|
| Stop sign  			| Stop sign, Priority road, Bicycles crossing, No vehicles, Yield							|
| No Passing		    | No Passing, No passing for heavy vehicles, Traffic signals, Priority road, HV prohibited	|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I didn't make it due that I have this 'tf_activation' is not defined error. Where to see the closed issues of the github repo?
