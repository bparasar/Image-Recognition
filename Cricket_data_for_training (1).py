#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:12:46 2018

@author: prayash
"""

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

#%%
#  data

path1 = '/Users/prayash/Downloads/Cricket_dataset/Cricket_bowlers'    #path of folder of images    
path2 = '/Users/prayash/Downloads/Cricket_dataset/Cricket_dataset_resized'  #path of folder to save images    

listing = os.listdir(path1) 
num_samples=size(listing)

print(num_samples)

for file in listing:
    if not file.startswith('.'):
        im = Image.open(path1 + '/' + file)   
        img = im.resize((img_rows,img_cols))
        gray = img.convert('L')
                #need to do some more processing here           
    gray.save(path2 +'/' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open('/Users/prayash/Downloads/Cricket_dataset/Cricket_dataset_resized' + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('/Users/prayash/Downloads/Cricket_dataset/Cricket_dataset_resized'+ '/' + im2)).flatten()
              for im2 in imlist],'f')
#immatrix.shape

    
    







            
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

img = tf.placeholder(tf.float32, shape =(441, 40000))

from keras.layers import Dense

# Keras layers can be called on TensorFlow tensors:
#x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
#x = Dense(128, activation='relu')(x)
#preds = Dense(5, activation='softmax')(x)

labels = tf.placeholder(tf.float32, shape=(441, 5))

#from keras.objectives import categorical_crossentropy
#loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init_op = tf.global_variables_initializer()
sess.run(init_op)

(img, labels) = (train_data[0],train_data[1])

with sess.as_default():
    for i in range(100):
        batch = train_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})







    
label=np.ones((num_samples,),dtype = int)
label[0:89]=0
label[89:179]=1
label[179:268]=2
label[268:356]=3
label[356:]=4

#label.shape

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img=immatrix[167].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

#%%

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 5
# number of epochs to train
nb_epoch = 70


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 441
X_test /= 441

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 100
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

#%%

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool),border_mode='same',
                        input_shape=(1, img_rows, img_cols)))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')


input_layer = tf.reshape(img, (441,1, 200, 200))

conv1 = tf.layers.conv2d(
      inputs= input_layer,
      filters=nb_filters,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)


pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,2],strides=2)

conv2 = tf.layers.conv2d(
      inputs= pool1,
      filters=nb_filters,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)

conv3 = tf.layers.conv2d(
      inputs= pool2,
      filters=nb_filters,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=2)

pool3_flat = tf.reshape(pool3, [441, 5*5*32])

dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

dropout = tf.layers.dropout(
      inputs=dense, rate=0.4)

logits = tf.layers.dense(inputs=dropout, units=5)

predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}







#%%

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1, validation_data=(X_test, Y_test))
            
            
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=1, validation_split=0.4)


# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print(plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['bmh'])





#%%       




#%%


# the input image

test_image = Image.open('/Users/prayash/Downloads/Zampa_test_image.jpg')
img_test = test_image.resize((img_rows,img_cols))
gray = img_test.convert('L')
img_test_array = array(gray)
m_test,n_test = img_test_array.shape[0:2]
plt.imshow(img_test_array)

immatrix_test = array(img_test_array.flatten(),'f')
immatrix_test.shape


img_test_bowler = immatrix_test.reshape(1, 1, img_rows, img_cols)
img_test_bowler = img_test_bowler.astype('float32')

y_pred_test = model.predict_classes(img_test_bowler)
print("And the player in the picture seems to be a :")
if y_pred_test == [0]:
    print("Legspinner")
elif y_pred_test == [1]:
    print("FastBowler")
elif y_pred_test == [2]:
    print("OffSpinner")
elif y_pred_test == [3]:
    print("SwingBowler")
elif y_pred_test == [4]:
    print("Slow-Arm-Chinamen")
    

p=model.predict_proba(immatrix_test)



input_image=X_train[3:,:,:,:]
print(input_image.shape)

plt.imshow(img_test_array,cmap ='gray')
plt.imshow(img_test_array)

X_test.shape
# Confusion Matrix

from sklearn.metrics import classification_report,confusion_matrix

y_pred = model.predict_classes(img_test_array)
print(y_pred)

p=model.predict_proba(input_image) # to predict probability

target_names = ['class 0(LegSpinner)', 'class 1(FastBowler)', 'class 2(Offspinner)','class 3(SwingBowler)','class 4(Slow-Arm-Chinamen)']
print(classification_report(np.argmax(y_pred), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

y_pred = model.predict_classes(input_image)
print(y_pred)

# saving weights

#fname = "weights-Test-CNN.hdf5"
#model.save_weights(fname,overwrite=True)



# Loading weights

#fname = "weights-Test-CNN.hdf5"
#model.load_weights(fname)

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import numpy as np
import cv2

cv2.__version__
help(cv2.ml)

#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap = cv2.VideoCapture('/Users/prayash/Downloads/Bowler_clip.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
 
    #net.setInput(blob)
    #detections = net.forward()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#!python /Users/prayash/Downloads/serving/tensorflow_serving/example/Cricket_model.py /tmp/cricket_bowler_model



    
import tensorflow as tf
 
#Prepare to feed input, i.e. feed_dict and placeholders

sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
saver_new2 = tf.train.Saver() 
#Now, save the graph
saver_new2.save(sess, '/Users/prayash/Downloads/CNN_tensorflow',global_step=1000)
 






import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.framework import graph_util

model_folder = '/Users/prayash/Downloads/CNN_tensorflow.py'

try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
except:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder)


output_node_names = "FullyConnected/softmax_tensor"

clear_devices = True

#saver = tf.train.Saver()

#saver.save(sess, '/Users/prayash/Downloads/Cricket_model')

saver = tf.train.import_meta_graph('/Users/prayash/Downloads/Cricket_model-1000.meta', clear_devices=clear_devices)



with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('/Users/prayash/Downloads/Cricket_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/Users/prayash/Downloads/'))












graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,                        # The session is used to retrieve the weights
            input_graph_def,             # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 
    

import numpy as np
import cv2 as cv


net = cv.dnn.readNetFromTensorflow('/Users/prayash/Downloads/prayash_cricket_model.pb','/Users/prayash/Downloads/graph.pbtxt')



image = cv2.imread(filename)
# Resizing the image to our desired size and
# preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)
 
 
graph = tf.get_default_graph()
 
y_pred = graph.get_tensor_by_name("y_pred:0")
 
## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 
 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)




