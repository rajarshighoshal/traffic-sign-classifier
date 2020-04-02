## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[//]: # (Image References)

[image1]: ./writeup_imgs/graph.png "Visualization"
[image2]: ./writeup_imgs/y_channel.png "Y channel extraction"
[image3]: ./writeup_imgs/augmentation.png "Augmentation"
[image4]: ./writeup_imgs/internet_imgs.png "External Images"
[image5]: ./external_imgs/tr2.png "Traffic Sign 2"
[image6]: ./external_imgs/tr3.png "Traffic Sign 3"
[image7]: ./external_imgs/tr4.png "Traffic Sign 4"
[image8]: ./external_imgs/tr5.png "Traffic Sign 5"
[image9]: ./external_imgs/tr6.png "Traffic Sign 6"

Overview
---
In this project, I have applied my learnings about deep neural networks and convolutional neural networks to classify traffic signs. I have trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I have tried out my model on images of German traffic signs that I have found on the web.


Data Set Summary & Exploration
---

#### 1. Basic Summery of Data

I have used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is diustributed across different classes of traffic signs:

![alt text][image1]

The average number of training examples per class is `809`, the minimum is `180` and the maximum `2010`, hence some labels are one order of magnitude more abundant than others.

Most common signs:
* `Speed limit (50km/h)` train samples: 2010
* `Speed limit (30km/h)` train samples: 1980
* `Yield` train samples: 1920
* `Priority road` train samples: 1890
* `Keep right` train samples: 1860

Most rare signs:
* `End of no passing` train sample: 210
* `End of no passing by vehicles over 3.5 metric tons` train sample: 210
* `End of all speed and passing limits` train sample: 210
* `Pedestrians` train sample: 210
* `Go straight or left` train sample: 180
* `Dangerous curve to the left` train sample: 180
* `Speed limit (20km/h)` train sample: 180


Design and Test a Model Architecture
---

#### 1. Data Preprocessing

Following the published [baseline model](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) on this problem I applied similar normalization and image enhancements. I have applied Gaussian Blur on the images to enhance sharpness. Then converted the images from RGB-space to YUV-space and extracted only the Y channel as using full YUV color space wasn't giving better results.

Here is some example of original vs transformed images:
![alt text][image2]

I decided to generate additional data using image augmentation because there wasn't much vartiety in the training images.

To add more data to the the data set, I used:-
* `Random rotation` of the images because it would help detect siogns even if the signs are rotated by some natural causes. 
* Apart from that I have also applied `random sacling` so that scaling doesn't have much effect on the effectiveness of the model.

Here is some example of original vs augmented images:

![alt text][image3]


#### 2. Final model architecture 

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Y channel image (from YVU space)      | 
| Convolution 3x3       | 1x1 stride, same padding, outputs 30x30x6     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 15x15x6                  |
| Convolution 3x3       | 1x1 stride, same padding, outputs 13x13x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 6x6x16                   |
| Convolution 3x3       | 1x1 stride, same padding, outputs 4x4x64      |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 2x2x64                   |
| Flattening            | Input 2x2x64, outputs 256                     |                        
| Fully Connected       | Input 256, outputs 128                        |
| RELU                  |                                               |
| Dropout               | Dropout with 50% probability of dropping out  |                                              
| Fully Connected       | Input 128, outputs 84                         | 
| RELU                  |                                               |
| Dropout               | Dropout with 50% probability of dropping out  |
| Fully Connected       | Input 84, outputs 43                          | 
| Softmax               |                                               |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

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
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image4] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

