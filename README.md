# **Traffic Sign Classifier** 

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
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/web_images.png "Traffic Signs"
[image5]: ./examples/histogram.png "Training Data Distribution"
[image6]: ./examples/traffic_signs.png "Traffic Signs"
[image7]: ./examples/normalize.png "Normalize"
[image8]: ./examples/fake.png "Fake"
[image9]: ./examples/histogram_fake.png "New Training Data Distribution"
[image10]: ./examples/augmented.png "Augmented"
[image11]: ./examples/web_images.png "Web Images"
[image12]: ./examples/softmax.png "Softmax"
[image13]: ./examples/featuremap1.png "Feature Map"
[image14]: ./examples/featuremap2.png "Feature Map"
[image15]: ./examples/featuremap3.png "Feature Map"

### Dataset
I trained the model using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Please [download the dataset here](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip), where the images have already been resized to 32x32. 

---
### Writeup / README

#### 1. Writeup / README

You're reading it! 

#### 2. Jupyter Notebook
Here's a link to my [project code](https://github.com/clarisli/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distribute over 43 classes. Each bin represents one class and its height indicates how many examples in that class.

![alt text][image5]

Here are the traffic sign images each with its corresponding class id.
![alt text][image6]

You can refer to [signnames.csv](./signnames.csv) for the name of each traffic sign.

### Design and Test a Model Architecture

#### 1. Pre-process the Data

As a first step, I decided to convert the images to grayscale for simplicity and data reduction. Color information is not a critical factor for traffic sign classification. Converting to grayscale reduces data size and training time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it ensures each pixel has similar data distribution. This helps reaching convergence faster while training the network.

![alt text][image7]

I decided to generate additional data because there are some classes under-represented in the dataset, I can make the model more robust and balanced by adding some randomly generated data into those minority classes.

To add more data to the the data set, I applied data augmentation to the dataset, randomly scale, shear, translate, and adjusting the brightness of an image. 

Here is an example of an original image and an augmented image:

![alt text][image10]

Following are some more examples on the fake images generated with random noise:

![alt text][image8]

The new training data distribution with fake images:

![alt text][image9]

#### 2. Model Achitecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x30 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x80      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x80 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x4x400      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x400 				|
| Fully connected		| input 1600, output 120        									|
| Fully connected		| input 120, output 84        									|
| Softmax				| input 84, output 43        									|
|						|												|
|						|												|
 


#### 3. Training the Model

To train the model, I used my local machine and AWS.

My final training parameters:
* EPOCHS = 30
* BATCH_SIZE = 128
* SIGMA = 0.1
* OPTIMIZER: AdamOptimizer (learning rate = 0.001)

#### 4. Approach for Finding a Solution

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 98.3% 
* test set accuracy of 97.1%

An iterative approach was chosen:
* I tried the classic LeNet as the initial architecture. I chose it because it is shallow and simple.
* LeNet turned out to be a good choice. I obtained a training accuracy 99% / validation 89% - since the training accuracy is much higher than validation, looks like it's overfitting.
* To overcome this, I decided to apply data augmentation technique to add more data and noise to the training set. Some classes are under-presented in the original training set, I made sure every class has at least 809(the mean of the training set) images. It helped to boost the model to a training accuracy 99.7% / validation 93.2% / test 91%.
* The model is still overfitting. I then added 2 layers of dropouts with a keep rate of 75% for the first layer, and 50% for the second. This is my first qualified solution, with a training accuracy 99.4% / validation 96.1% / test 93%. 
* The test accuracy still needs improvement. I figured it's the the time to add an extra layer to my net because the model might need more parameters to gain a better representation of the traffic signs. I abtained training accuracy 99.9% / validation 96.7%, and test 94.2%.
* The goal was to increase the validation and test accuracy, I generated more random data to the training set and increased the training set to 3000 examples per class and trained for 30 epoch. A final result of training accuracy 99.9% / validation 98.3% / test 97.1% was obtained.
* In additional to above, I also tuned other parameters such as the randomness in augmented data, the keep probability of dropout, learning rate, stride size, number of neurons ... etc. 
 

### Test on New Images

#### 1. Traffic signs found on the web.

Here are five German traffic signs that I found on the web:

![alt text][image11]

The road work image might be difficult to classify because it has a more complex background.

#### 2. Accuracy

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)    		| Speed limit (70km/h)  									| 
| Slippery road     			| 	Slippery road 										|
| Road work					| Beware of ice/snow											|
| Children crossing	      		| Children crossing					 				|
| Wild animals crossing			| Wild animals crossing      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Not surprisingly, it's much lower than the test accuracy obtained above. It's hard to obtain high accuracy with only 5 images in the test set.

#### 3. Softmax Probabilities

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![alt text][image12]

### Visualizing the Neural Network
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image13]
![alt text][image14]
![alt text][image15]

It looks like the first layer is detecting the edges and shape, and hard to tell what characteristics it uses for the deeper layers.
