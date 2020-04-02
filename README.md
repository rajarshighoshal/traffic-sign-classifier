## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[//]: # (Image References)

[image1]: ./writeup_imgs/graph.png "Visualization"
[image2]: ./writeup_imgs/y_channel.png "Y channel extraction"
[image3]: ./writeup_imgs/augmentation.png "Augmentation"
[image4]: ./writeup_imgs/internet_imgs.png "External Images"
[image5]: ./writeup_imgs/predictions.png "Predictions on External Images"


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


#### 3. Model Hyperparameters

To train the model, I have choosed the folowing hyperparameters:
* Learning Rate - `0.0001`
* Batch Size - `32`
* No. of Epochs - `300`

#### 4. Model Building Approach

* At first I have used classic `LeNet` architecture. 
* The initial LeNet model was overfitting or underfitting, even after tuning the hyperparameters in various ways I was unable to get better results.
*  As a result I have decided to modify it.
* I have changed the LeNet's 5x5 kernel size for convolution layers to 3x3 kernal size. 
* I have also added another convolution layer followed by a maxpooling layer to increase the depth of the feature map. 
* I have also added one additional hidden layer in the fully connected layers to decrease the neuron sizes more gradually.
* Apart from that in between each fully connected layer I have decided to add a dropout layer with a dropout probability of 50% to stop overfitting.
* As the optimiser I have choosed Adam optimiser.
* After much testing I have decided to keep learning rate `0.0001` and decraes the batch size to `32`; also I have choosed the epoch as `300` as the model tarining and valkidation accuracy was continuing to increase for any number below that.

My final model results were:
* training set accuracy of `1.00`
* validation set accuracy of `0.96`
* test set accuracy of `0.95`

### Test the Model on New Images

#### 1. Found six German traffic signs found on the web to try out the model

Here are six German traffic signs that I found on the web:

![alt text][image4] 

The images might be difficult to classify because they are of various sizes. Some images are as big as 256x256 whereas some others are much much smaller as 8x8. As a result some images might contain more minute details which aren't present in the training images or more noises which is not also a part of training images.

#### 2. Predict the sings using the model

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)                          | 
| Roundabout mandatory  | Roundabout mandatory                          |
| Ahead only            | Ahead only                                    |
| No vehicles           | No vehicles                                   |
| Go straight or left   | Go straight or left                           |
| General caution       | General caution                               |    

Here is the output from the model as prediction with respect to the external images:

![alt text][image5]


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Model confidence for the new images

The softmax probabilities for the top 5 classes for image is given below:
* Top 5 Labels for image `Speed limit (30km/h)`:
     - `Speed limit (30km/h)` with prob = 1.00 
     - `Speed limit (50km/h)` with prob = 0.00 
     - `Speed limit (70km/h)` with prob = 0.00 
     - `Speed limit (20km/h)` with prob = 0.00 
     - `Speed limit (80km/h)` with prob = 0.00 
* Top 5 Labels for image `Roundabout mandatory`:
     - `Roundabout mandatory` with prob = 0.95 
     - `No entry` with prob = 0.04 
     - `Keep right` with prob = 0.00 
     - `Turn right ahead` with prob = 0.00 
     - `Keep left` with prob = 0.00 
* Top 5 Labels for image `Ahead only`:
     - `Ahead only` with prob = 1.00 
     - `Go straight or right` with prob = 0.00 
     - `Turn left ahead` with prob = 0.00 
     - `Turn right ahead` with prob = 0.00 
     - `Speed limit (60km/h)` with prob = 0.00 
* Top 5 Labels for image `No vehicles`:
     - `No vehicles' with prob = 0.94 
     - `Priority road` with prob = 0.06 
     - `Yield` with prob = 0.00 
     - `Ahead only` with prob = 0.00 
     - `Keep right` with prob = 0.00 
* Top 5 Labels for image `Go straight or left`:
     - `Go straight or left` with prob = 1.00 
     - `Roundabout mandatory` with prob = 0.00 
     - `Keep right` with prob = 0.00 
     - `Turn left ahead` with prob = 0.00 
     - `Ahead only` with prob = 0.00 
* Top 5 Labels for image `General caution`:
     - `General caution` with prob = 1.00 
     - `End of no passing by vehicles over 3.5 metric tons` with prob = 0.00 
     - `No passing for vehicles over 3.5 metric tons` with prob = 0.00 
     - `No entry` with prob = 0.00 
     - `Vehicles over 3.5 metric tons prohibited` with prob = 0.00

It can be observed from the above data that model is very confident with each of the outcomes. The reason might be that the model has received almost similar kinds of inputs on which it was trained. No images was too much rotated, neither any of them cobntains any bad lighting condition or any additional objects other than traffic signs. But that aside, it cxan be concluded that if the model recives almost similar images on which it was trained it will give very good results on totally unseen images.


Dependencies
---
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.