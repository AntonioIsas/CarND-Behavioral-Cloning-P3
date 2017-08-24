# **Behavioral Cloning**

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[center]: ./examples/center.jpg "Center Image"
[left]: ./examples/left.jpg "Left Image"
[right]: ./examples/right.jpg "Right Image"
[flipped]: ./examples/flipped.jpg "Center Image - Flipped"
[hls]: ./examples/hls.jpg "HLS Image"
[cropped]: ./examples/cropped.jpg "Cropped Image"
[loss]: ./examples/figure_1.png  "Loss"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First of all my model preprocess the images by cropping some of the extra portions in the image (code line 74), Then I use a Keras Lambda layer to normalize the input (code line 75)

The model I'm using is the NVidia Architecture, it consists of 3 convolution layers of 5x5 filter and depths 24, 36 and 48 (code lines 78-80)

Followed by two convolution layers with 3x3 filter and depth of 64 (code lines 81, 82)

The 5 convolution layers are using RELU for activation to introduce nonlinearity

Then I use 3 fully connected layers of size 100, 50 and 10

and a final fully connected layer that provides the output with the steering angle

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 23, 69,70 and 92). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).
I decided to train for 5 epochs

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving, with multiple cameras and a correction of 0.1

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to base on existing and proven networks and tune them as needed

My first step was to use a convolution neural network model similar to the LeNet architecture, I thought this model might be appropriate because it works well to identify features in the images

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and on the validation set.

I run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I decided to change to a more powerful model and chose the NVIDIA architecture, I also augmented the data using the other cameras.

After this training and some more tests the model was able to complete the first track but was not doing very well generalizing for the second one so I added more data from the second one and kept tuning.

The model was doing well on both tracks, however when adding some details like the shadows it would run out off track on both. I realized the initial data didn't had any shadows so I added some more data as well as trying to use a different color space.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road. It still struggles with one of the curves of the second track when it has shadows but manages to stay on the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74-89) consisted of a convolution neural network with the following layers and layer sizes:

1. Cropping layer receives an input of 160,320,3 removes 60 pixels from top and 20 from bottom
2. Lambda layer for normalizing
3. Convolution Layer size 5x5 and 24 filter
4. Convolution Layer size 5x5 and 36 filter
5. Convolution Layer size 5x5 and 48 filter
6. Convolution Layer size 3x3 and 64 filter
7. Convolution Layer size 3x3 and 64 filter
8. Flatten Layer
9. Fully connected 100
10. Fully connected 50
11. Fully connected 10
12. Fully connected 1 (output)

It uses Relu activation for each convolution layer, and minimizes the mean squared error with an adam optimizer.

![alt text][loss]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two normal laps and two laps in reverse on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I tried to use recoveries but the data was giving poor results, so instead I used the left and right cameras, Here is the left and right cameras from the image above

![alt text][left]
![alt text][right]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help generalize the model by adding a new track to it
For example, here is an image that has then been flipped:

![alt text][center]
![alt text][flipped]

After the collection process, I randomly shuffled the data set and put 20% of the data into a validation set. I then preprocessed this data by converting it to HLS color space

![alt text][hls]

Then the model crops and normalizes the image

![alt text][cropped]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
