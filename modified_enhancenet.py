#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from nnlib import *
#%matplotlib inline

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.chdir("/content/drive/App/EnhanceNet")

#import cv2
from PIL import Image
image_files = os.listdir("/content/drive/App/CVAE/natural_images")
trainX = []
# print(len(image_files))
for i, file in enumerate(image_files):
  if i%100 == 0 and file.endswith(".jpg"):
    img = Image.open("/content/drive/App/CVAE/natural_images/"+file).convert('RGB')
    w, h = img.size
#   ar = cv2.cvtColor(cv2.imread("/content/drive/App/CVAE/natural_images/"+file),cv2.COLOR_BGR2RGB)
    trainX.append(np.array(img)/255)
trainX = np.asarray(trainX)
#trainX = trainX/255
#plt.axis("off")
#plt.imshow(trainX[0])

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(trainX, test_size = 0.1, random_state=2)

###############################################################################################
import tensorflow.contrib.layers as lays
def autoencoder(inputs):
  with tf.variable_scope("CAE", reuse = tf.AUTO_REUSE) :
    net = lays.conv2d(inputs, 3, [3, 3], stride=2, padding='SAME')
    net = lays.conv2d(net, 3, [3, 3], stride=2, padding='SAME')
    net0 = lays.conv2d(net, 3, [3, 3], stride=2, padding='SAME')
    #net = lays.conv2d(net, 3, [3, 3], stride=2, padding='SAME')
    
    upsampled256 = tf.image.resize_bicubic(net, (256,256))
    upsampled128 = tf.image.resize_bicubic(net, (128,128))
    upsampled64 = tf.image.resize_bicubic(net, (64,64))

    net1 = lays.conv2d_transpose(net0, 3, [3, 3], stride=2, padding='SAME')
    net1 = tf.add(net1,upsampled64)
    net2 = lays.conv2d_transpose(net1, 3, [3, 3], stride=2, padding='SAME')
    net2 = tf.add(net2,upsampled128)
    net3 = lays.conv2d_transpose(net2, 3, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
    net3 = tf.add(net3,upsampled256)
  return [net3,net1]

ae_inputs = tf.placeholder(tf.float32, (None, 256, 256, 3))
ae_outputs,ae_LR_outputs = autoencoder(ae_inputs)       #tf.image.resize_bicubic(temp, (256,256))
#loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))
#train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init = tf.global_variables_initializer()
#################################
'''all_vars = tf.global_variables()
model_one_vars = [k for k in all_vars if k.name.startswith("CAE")]
print(len(all_vars))
print(len(model_one_vars))'''
#################################
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)
  saver.restore(sess, "/content/drive/App/EnhanceNet/models/model.ckpt")
  #batch_img = x_test
  recon_LR_img, recon_HR_img = sess.run([ae_LR_outputs, ae_outputs], feed_dict={ae_inputs: x_test})
###############################################################################################
PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
for i, image in enumerate(recon_LR_img[:15]) :
    imgs = np.expand_dims(image, axis=0)
    imgsize = np.shape(imgs)[1:]
    #print('processing %s' % fn)
    xs = tf.placeholder(tf.float32, [1, imgsize[0], imgsize[1], imgsize[2]])
    rblock = [resi, [[conv], [relu], [conv]]]
    ys_est = NN('generator',
                [xs,
                 [conv], [relu],
                 rblock, rblock, rblock, rblock, rblock,
                 rblock, rblock, rblock, rblock, rblock,
                 [upsample], [conv], [relu],
                 [upsample], [conv], [relu],
                 [conv], [relu],
                 [conv, 3]])
    ys_res = tf.image.resize_images(xs, [4*imgsize[0], 4*imgsize[1]],
                                    method=tf.image.ResizeMethod.BICUBIC)
    ys_est += ys_res + PER_CHANNEL_MEANS
    sess = tf.InteractiveSession()
    ###################################################################
    all_vars = tf.global_variables()
    model_one_vars = [k for k in all_vars if k.name.startswith("CAE")]
    set_1 = set(model_one_vars)
    temp_set = [k for k in all_vars if k.name.startswith("generator")]
    model_two_vars = [o for o in all_vars if o not in set_1]
    '''print(len(all_vars))
    print(len(model_one_vars))
    print(len(temp_set))
    print(len(model_two_vars))'''
    #for i in model_one_vars :
      #print(i)
    
    ###################################################################
    tf.train.Saver(model_two_vars).restore(sess, os.getcwd()+'/weights')
    output = sess.run([ys_est, ys_res+PER_CHANNEL_MEANS],
                      feed_dict={xs: imgs-PER_CHANNEL_MEANS})
    #plt.axis("off")
    #plt.imshow(output[0][0])
    saveimg(output[0][0], 'output/IMG-%d-EnhanceNet.png' % i)
    saveimg(recon_HR_img[i], 'output/IMG-%d-HR.png' % i)
    saveimg(x_test[i], 'output/IMG-%d-LR.png' % i)
    #halter = input("halted")
    #saveimg(output[1][0], 'output/%s-Bicubic.png' % fne)
    sess.close()
    tf.reset_default_graph()

