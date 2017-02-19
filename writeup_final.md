#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/placeholder.png "Model Visualization"
[image2]: ./output_images/center_lane_driving.jpg "Grayscaling"
[image3]: ./output_images/recovery_from_left.jpg "Recovery Image"
[image4]: ./output_images/recovery_from_right.jpg "Recovery Image"
[image5]: ./output_images/recovery_in_turn.jpg "Recovery Image"
[image6]: ./output_images/hold_during_apex.jpg "Normal Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

##### Early observations
In the spirit of truly applying transfer learning (and drawing inspiration from the paper 'learning how to learn by gradient descent by gradient descent', 
I decided to learn a suitable model architecture by observing as many comparable solutions as possible. I was not interested in the details at first, but merely observed the overarching structure.
The first step was to evaluate the Nvidia approach, which uses a series of 5 convolution layers with RELU activation, followed by a flatten layer, then alternating dropout and fully connected layers.
Next, I looked at the comma.ai approach, which was similar, but used less convolutions, at 5x5 then 3x3 compared to nVidia's 8x8 to 4x4 to 2x2.

These provided a good idea of the general direction, so I implemented a simple version of these at first. I wanted to keep it simple in order to debug more easily, and train faster.

    My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

##### First model attempt
My first architecture was a simplification of the comma.ai model, dubbed 'JAMES Mai':

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(ELU())
model.add(Dense(512))
model.add(ELU())
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

##### Evaluation
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I trained for 3 epochs and evaluated the mean squared error.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. 
This implied that the model was overfitting. 
This was further proven in the simulator, where the model was swerving a lot, failing to compensate for turns correctly, and crashing. Thus earning the name "JAMES Mai"
To combat the overfitting, I initially modified the model so that it had dropout, similar to the actual comma.ai system. However, this did not improve the system greatly.
Then I observed (in the discussion forums) a lot of people with overfitting models, despite very different architectures, which implied the issue was moreso around data preparation.

##### Data processing focus
I then implemented some data processing steps, detailed in section three. These predominantly included the omission of many frames with low steering angles.
When the data was augmented, normalized and ultimately more adequate prepared for combatting overfitting, I tried training again with more epochs and no dropout. 
This provided a much stronger model which completed the training track in a drunken stupor. However, it was still failing on the challenge track. 
To improve the driving behavior in these cases, I focused on data augmentation again and added data of a real race line. 

##### Input smoothing
I also added some input smoothing in drive.py, which definitely helped maintain a smooth line around the track.
In this case, I merely used `smoothed_angle = current_angle_prediction * alpha + last_smoothed_angle * (1-alpha) `.
An alpha of 0.5 was enough to smooth the line without ruining the responsiveness in sharp turns.
At the end of the process, the vehicle is able to drive autonomously around the training track without leaving the road. 
It can also complete the challenge track at max speed, with only a minor bumper scrape. 

####2. Final Model Architecture

The final model architecture (model.py lines 28-52) consisted of a convolution neural network with the following layers and layer sizes:

_______________________________________________________________
A Lambda layer with an input shape of (66,200,3).
5x5 Convolutional layer with a depth of 24 and ELU activation.
5x5 Convolutional layer with a depth of 36 and ELU activation.
5x5 Convolutional layer with a depth of 48 and ELU activation.
3x3 Convolutional layer with a depth of 64 and ELU activation.
3x3 Convolutional layer with a depth of 64 and ELU activation.
A flatten layer.
A fully-connected layer with a depth of 128 and ELU activation.
A fully-connected layer with a depth of 32 and ELU activation.
A fully-connected layer with a depth of 8 and ELU activation.
A fully connected layer with a depth of 1.
_______________________________________________________________

##### Notes on final model
Dropout was used intermittently during development, but ultimately my specific data pruning techniques were more effective than random dropout.
The Lambda layer maintained the same factor as the comma.ai model, but a new data shape.
After data pruning and angle-distribution normalization was shown to be very effective, I added on layers to better generalize behavior.
This came at the cost of training time, but it's ultimately still a fairly low-parameter network compared to what some people are using.


####3. Creation of the Training Set & Training Process

The data collection phase was ultimately the most important factor in the final driving ability. 
I recorded a full lap on the basic track counterclockwise, then clockwise, driving mostly in the center. 
Here is an example image of center lane driving:

![center driving][image2]

This resulted in a model which overfit, and would sway side to side drastically before running off the road. 
I then recorded a series of recoveries from edge positions. This helped decrease the worst of the swaying.
These images show what a recovery looks like starting from the left and right :

![recovery from left][image3]
![recovery from right][image4]

However, after training on this data, the model was still weak in corners. 
If it ended up on the inside edge of a corner, it would turn away aggressively, and be unable to compensate going off the road.
I then recorded the vehicle recovering from the left side and right sides of the road during a curve.
This was so the vehicle would learn to not rely excusively on the centering, but hopefully, also the curvature of the road.

![Recovery into a curve][image5]

As mentioned later, to beat the challenge tracks I also ended up recording some positions where the opposite behavior was warranted.
Namely, when holding a trajectory into the apex of a turn. This behavior is actually counter-intuitive from the previous training.
This image shows an example where the car needs to rely on the curvature of the track to outweight the car's position in the track.

![Holding into an apex][image6]

I had doubts that this behavior would generalize, but it worked beautifully, and I eagerly renamed my model SEBASTaiN LOEB.
Then I repeated this process on track two in order to get more data points, and again on the beta tracks.
The beta simulator's second track was quite tough- having an xbox controller with joysticks made life a lot easier. 
Using all the available tracks from both sims seemed to really help the system generalize. 
Ultimately,after the collection process, I had 31215 data points. 

##### Processing
I then preprocessed this data by shifting the brightness and horizon of the images. 
I also partially shadowed some images to ensure it wasn't learning brightness.
I flipped images with a steering angle over 0.3, thinking that this would enhance the models response during cornering.
I didn't save any, so let your imagination run wild on this one.
I then eliminated a ton of straight line images, as mentioned before, via binning them by angle and randomly deleting samples from the most over-represented bins.
I finally randomly shuffled the data set and put 5% of the data into a validation set. 

##### Training
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 5 as evidenced by the driving performance. The loss continues to decrease with more epochs, but it overfits. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.

This gave solid results on the first track, very smooth driving and no close calls.
But I also noticed that the car was couldn't get through the challenge tracks, mostly because of the sharp turns.

As an avid sim racer, this would not stand- so I augmented my data set again with the best racing line I could muster.
I practiced the courses a few times, and emphasised wide entries into the apex, and extremely smooth steering.

After adding these data points and re-training, the car performed beautifully on the alpha challenge track, making it through at high speed.
It actually replicated my driving line more closely- though not close enough to scare anyone, but you could notice it hug the apex a bit more.

I haven't been able to complete the beta challenge track- I might just need more data but the aggressive hills make it quite difficult to get a decent view of the road.


##### Conclusion

This project was a blast. I was very reliant on intuition to make a working pipeline, but I was impressed by how often intuition worked for this.
Neural networks are a black box, so the relationship between the 'engineer and engine' here is definitely intimate. 
It's definitely important to grasp the fundamentals, but also use what works. As Newton said, "stand on the shoulders of giants"
I've never been more excited by any program than when my model directly adapted to my race line, which is what I only had a gut feeling it would do.

In the future, I'll definitely put more early effort into data augmentation and normalizing the data set. 
Also, I'd like to expound on the non-NN driving factor, such as the steering input smoothing- I think this could be much better.
This project gave me a good feel for 'Garbage in, garbage out', and also emphasized that size (of the data set) isn't everything.

Looking forward to more of this next term!

