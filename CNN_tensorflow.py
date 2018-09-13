#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:37:04 2018

@author: prayash
"""

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
from keras.layers import Dense
from keras import backend as K              
import tensorflow as tf


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

sess = tf.Session()

K.set_session(sess)

img = tf.placeholder(tf.float32, shape =(441, 40000))
labels = tf.placeholder(tf.float32, shape=(441, 5))

init_op = tf.global_variables_initializer()
sess.run(init_op)

label=np.ones((num_samples,),dtype = int)
label[0:89]=0
label[89:179]=1
label[179:268]=2
label[268:356]=3
label[356:]=4

#label.shape

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

(img, labels) = (train_data[0],train_data[1])



def cnn_model_fn(features,labels,mode):
    input_layer = tf.reshape(features["img"], (-1,200, 200,1))

    conv1 = tf.layers.conv2d(
            inputs= input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)


    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,2],strides=2)

    conv2 = tf.layers.conv2d(
            inputs= pool1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)

    conv3 = tf.layers.conv2d(
            inputs= pool2,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=2)

    pool3_flat = tf.reshape(pool3, [50, 625*32])

    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4)

    logits = tf.layers.dense(inputs=dropout, units=5,name = "softmax")

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
          }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            loss=loss, eval_metric_ops=eval_metric_ops)
            
bowler_classifier = tf.estimator.Estimator(
    model_fn= cnn_model_fn, model_dir="Users/prayash/Downloads/cricket_CNN_model")


tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

#estimator = tf.estimator.DNNClassifier(feature_columns=['LegSpinner', 'FastBowler', 'Offspinner','SwingBowler','Slow-Arm-Chinamen'],hidden_units=[1024, 512, 256],warm_start_from="./")

train_input_fn =tf.estimator.inputs.numpy_input_fn(
    x={"img": img},
    y=labels,
    batch_size=50,
    num_epochs= 70,
    shuffle=True,
    )

bowler_classifier.train(
    input_fn=train_input_fn,
    steps= 2000,
    hooks=[logging_hook])



onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)













