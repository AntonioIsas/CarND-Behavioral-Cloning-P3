import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
meassurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    meassurement = float(line[3])
    meassurements.append(meassurement)

augmented_images, augmented_meassurements = [], []
for image, meassurement in zip(images, meassurements):
    #Add normal image
    augmented_images.append(image)
    augmented_meassurements.append(meassurement)
    #Add flipped image
    augmented_images.append( cv2.flip(image,1) )
    augmented_meassurements.append(meassurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_meassurements)


model = Sequential()
#Normalize
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#LeNet
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
