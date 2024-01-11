import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import the training data
train_normal_data = os.listdir("../../../Datasets/projects/chest_xray/train/NORMAL")
train_pneu_data = os.listdir("../../../Datasets/projects/chest_xray/train/PNEUMONIA")

print('Normal: ', len(train_normal_data))
print('Pneumonia: ', len(train_pneu_data))

# create labels for the data
# normal -> 0
# pneumonia -> 1

normal_labels = [0]*len(train_normal_data)
pneu_labels = [1]*len(train_pneu_data)

print('Normal labels: ', len(normal_labels))
print('Pneumonia labels: ', len(pneu_labels))

labels = normal_labels + pneu_labels

# # observe the first image from the normal dataset
# image=mpimg.imread("chest_xray/train/NORMAL/" + train_normal_data[0])
# plt.imshow(image)

# plt.show()

# # observe the first image from the pneu dataset
# image=mpimg.imread("chest_xray/train/PNEUMONIA/" + train_pneu_data[0])
# plt.imshow(image)

# plt.show()

# Image Processing
normal_path = "../../../Datasets/projects/chest_xray/train/NORMAL/"
pneu_path = "../../../Datasets/projects/chest_xray/train/PNEUMONIA/"
img_data = []

for img_file in train_normal_data:
    image = Image.open(normal_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    img_data.append(image)

for img_file in train_pneu_data:
    image = Image.open(pneu_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    img_data.append(image)

print("Num of images: ", len(img_data))

# import the training data
test_normal_data = os.listdir("../../../Datasets/projects/chest_xray/test/NORMAL")
test_pneu_data = os.listdir("../../../Datasets/projects/chest_xray/test/PNEUMONIA")
test_normal_path = "../../../Datasets/projects/chest_xray/test/NORMAL/"
test_pneu_path = "../../../Datasets/projects/chest_xray/test/PNEUMONIA/"
test_img_data = []

for img_file in test_normal_data:
    image = Image.open(test_normal_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    test_img_data.append(image)

for img_file in test_pneu_data:
    image = Image.open(test_pneu_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    test_img_data.append(image)

test_normal_labels = [0]*len(test_normal_data)
test_pneu_labels = [1]*len(test_pneu_data)

test_labels = test_pneu_labels + test_normal_labels
Y_test = np.array(test_labels)

print("Num of images: ", len(img_data))
print("Num of test images: ", len(test_img_data))

# Converting data and lables to numpy array
X = np.array(img_data)
Y = np.array(labels)

print(X[0])

X_train = X/255
X_test = np.array(test_img_data)/255

# Create CNN
import tensorflow as tf 
from tensorflow import keras 

num_of_classes = 2
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3),activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)



history=model.fit(X_train,Y, validation_split=0.1, verbose=1, epochs=5)
loss,accuracy=model.evaluate(X_test, Y_test)


plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')



plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')

