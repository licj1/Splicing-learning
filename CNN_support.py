# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:14:43 2019

@author: pc
"""
import support_based as spb
import CNN_function as cf
import numpy as np
import tensorflow as tf


# CNN model
def CNNnet(tr_data, tr_label, te_data, te_label, sam_num, n_way, com_label_str, com_img_str, str1 = 'none', iter_num=5000):
    # clean graph
    tf.reset_default_graph()
    # cnn framework
    x, y_, is_train, y_conv, loss, accuracy, train_step = cf.Net(sam_num, n_way, com_label_str, com_img_str)
    # initialize variables
    init = tf.global_variables_initializer()
    # step4: train
    # run and train
    with tf.Session() as sess:
        sess.run(init)
        bat_num = 200
        tr_result = []
        te_result = []
        global_result = []
        break_num = False
        # get the training index
        tr_index = spb.com_train_data(tr_label, sam_num, com_label_str)
        ind = 0
        for i in range(iter_num):
            bat_in, bat_out, ind = spb.get_train_data_depth_concat(ind, bat_num, sam_num, n_way, com_label_str, com_img_str, tr_index, tr_data, tr_label)
            optim, tr_loss, tr_acc, tr_y = sess.run([train_step, loss, accuracy, y_conv], feed_dict={x: bat_in, y_: bat_out, is_train: True})
            tr_result.append(np.asarray([i, tr_loss, tr_acc, bat_out.shape[0]]))
            
            # get the training global loss
            global_result = spb.get_global_loss(tr_result, tr_index)
            
            # whther stop
            if i > 100 and len(global_result)>2 and global_result[-1] >= global_result[-2]:
                break_num = True
            
            # test
            if i % 50 == 0 or break_num:
                bat_in, bat_out = spb.get_test_data_depth_concat(sam_num, n_way, com_label_str, com_img_str, te_data, te_label)
                te_loss, te_acc, te_y = sess.run([loss, accuracy, y_conv], feed_dict={x: bat_in, y_: bat_out, is_train: False})
                te_acc1 = te_acc
                if com_label_str == 'all':
                    te_acc1 = spb.anal_res(te_label, te_y, sam_num, n_way)
                te_result.append(np.asarray([i, te_loss, te_acc, te_acc1]))
                print(i, tr_loss, tr_acc, '||', te_acc, te_acc1)
                #print(datetime.datetime.now())
                
            # stop
            if break_num:
                break
        
        result = [np.stack(tr_result, 0), global_result, np.stack(te_result, 0)]
        # save the result
        spb.save_result(result, str1 )

    return    
    
    
    










