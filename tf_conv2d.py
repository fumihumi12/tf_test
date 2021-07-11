#
# tf_conv2d.py
# 2021-07-11 N.Kashiyama

import tensorflow as tf
import numpy as np

x_in=np.array([[
	[[1],[2],[3],[4],[5]],
	[[6],[7],[8],[9],[10]],
	[[11],[12],[13],[14],[15]],
	[[16],[17],[18],[19],[20]],
	[[21],[22],[23],[24],[25]],]])
x=tf.constant(x_in, dtype=tf.float32)
print('x:',x.shape)

y_in=np.array([[
	[[1],[2],[3],[4],[5]],
	[[6],[7],[8],[9],[10]],
	[[11],[12],[13],[14],[15]],
	[[16],[17],[18],[19],[20]],
	[[21],[22],[23],[24],[25]],]])
y=tf.constant(y_in, dtype=tf.float32)
print('y:',y.shape)

kernel_in=np.array([
	[[[0]],[[1]]],
	[[[2]],[[3]]],])
kernel=tf.constant(kernel_in, dtype=tf.float32)
print('kernel:',kernel.shape)

x_conv = tf.nn.conv2d(x, kernel, strides=[1,1,1,1], padding='VALID')
print('conv:',x_conv.shape)
print(x_conv)

y_conv = tf.nn.conv2d_transpose(x_conv, kernel, [1,5,5,1], strides=[1,1,1,1], padding='VALID')
print('trans:',y_conv.shape)
print(y_conv)
