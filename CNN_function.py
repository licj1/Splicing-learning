# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:37:20 2019

@author: pc
"""
import tensorflow as tf


# CNN framework for all samples combining
def Net(sam_num, n_way, com_label_str, com_img_str):
    # check the combining way
    if com_label_str == 'all':
        cat_num = n_way**sam_num
    elif com_label_str == 'cat':
        cat_num = n_way
    
    # place holder
    y_ = tf.placeholder(tf.float32, [None, cat_num])
    is_train=tf.placeholder(tf.bool)
    
    # CNN conv and max layers
    if com_img_str == 'depth':
        x = tf.placeholder(tf.float32, [None, 28, 28, sam_num])
        net = net_depth(x, is_train, sam_num)
    elif com_img_str == 'width':
        x = tf.placeholder(tf.float32, [None, 28, 28*sam_num, 1])
        net = net_width(x, is_train, sam_num)
    elif com_img_str == 'width_gap':
        x = tf.placeholder(tf.float32, [None, 28, 28*sam_num + 3*(sam_num -1), 1])
        net = net_width(x, is_train, sam_num)
    
    # full connectivity layers
    w_fc2 = weight_variable([64, cat_num])
    b_fc2 = bias_variable([cat_num])
    y_conv = tf.nn.softmax(tf.matmul(net, w_fc2) + b_fc2)

    # min the loss function
    diff=tf.losses.softmax_cross_entropy(onehot_labels = y_, logits = y_conv)
    loss = tf.reduce_mean(diff)
    
    # ckeck the predicted result and the real label
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    return x, y_, is_train, y_conv, loss, accuracy, train_step
    
# depth concat
def net_depth(x, is_train, sam_num):
    net = conv_layer(x, [3, 3, sam_num, 64], 1, is_train)
    net = max_pools(net, 2, 2)
    net = conv_layer(net, [3, 3, 64, 64], 1, is_train)
    net = max_pools(net, 2, 2)
    net = conv_layer(net, [3, 3, 64, 64], 1, is_train)
    net = max_pools(net, 2, 2)
    net = conv_layer(net, [3, 3, 64, 64], 1, is_train)
    net = avg_pools(net, 3, 3, 1)
    net = tf.reshape(net, [-1, 64])
    
    return net

# width concat
def net_width(x, is_train, sam_num):
    net = conv_layer(x, [3, 3, 1, 64], 1, is_train)
    net = max_pools(net, 2, 2)
    net = conv_layer(net, [3, 3, 64, 64], 1, is_train)
    net = max_pools(net, 2, 2)
    net = conv_layer(net, [3, 3, 64, 64], 1, is_train)
    net = max_pools(net, 2, 2)
    net = conv_layer(net, [3, 3, 64, 64], 1, is_train)
    net = avg_pools(net, 3, net.get_shape()[2], 1)
    net = tf.reshape(net, [-1, 64])
    
    return net
       
# cnn function --------------------------------------------------------------------------------
def conv_layer(inp, shape, stride, is_train):
    w = weight_variable(shape)
    inp=conv2d(inp, w, stride)
    inp = tf.layers.batch_normalization(inp, training = is_train)
    outp = tf.nn.relu(inp)
    return outp

def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))

def conv2d(a, w, s):
    return tf.nn.conv2d(a, w, strides=[1, s, s, 1], padding='SAME')
    
def max_pools(a, h, s):
    return tf.nn.max_pool(a, ksize=[1, h, h, 1], strides=[1, s, s, 1], padding="VALID")

def avg_pools(a, h1, h2, s):
    return tf.nn.avg_pool(a, ksize=[1, h1, h2, 1], strides=[1, s, s, 1], padding="VALID")

# --------------------------------------------------------------------------------------------