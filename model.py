import csv
import cv2
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import matplotlib.pyplot as plt

data_name = 'data'
samples = []
with open(data_name+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:
                # create adjusted steering measurements for the side camera images
                correction = 0.1 # this is a parameter to tune
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                directory = data_name+"/IMG/" # fill in the path to your training IMG directory
                img_center = cv2.cvtColor(cv2.imread(directory + batch_sample[0].split('/')[-1]), cv2.COLOR_BGR2HLS)
                img_left = cv2.cvtColor(cv2.imread(directory + batch_sample[1].split('/')[-1]), cv2.COLOR_BGR2HLS)
                img_right = cv2.cvtColor(cv2.imread(directory + batch_sample[2].split('/')[-1]), cv2.COLOR_BGR2HLS)

                # Augment data ( mirrored images )
                flip_img_center = cv2.flip(img_center,1)
                flip_img_left = cv2.flip(img_left,1)
                flip_img_right = cv2.flip(img_right,1)

                flip_steering_center = steering_center*-1.0
                flip_steering_left = steering_left*-1.0
                flip_steering_right = steering_right*-1.0

                # add images and angles to data set
                images.extend([img_center, img_left, img_right,
                              flip_img_center, flip_img_left, flip_img_right])
                angles.extend([steering_center, steering_left, steering_right,
                              flip_steering_center, flip_steering_left, flip_steering_right])

            # convert data to numpy arrays
            X_train = np.array(images)

            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Nvidia
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

# output
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3*2,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*3*2,
                    nb_epoch=5)

model.save('model_test.h5')
print("Model saved")

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
