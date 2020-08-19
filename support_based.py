# basic tools

import numpy as np
import pickle
import os
import string
import random

# This function is used to load dataset
def load_data():
    with open('/project/huangli/Project/Project5/MNIST_model/MNIST_data_np/mnist_data_list.txt', 'rb') as fi:
        data = pickle.load(fi)
    return data[0], data[1], data[2], data[3]


# this function is used to select specific number of samples from dataset
def select_samples(tr_data, tr_label, te_data, te_label, n_way, n_shot):
    # disorder dataset
    index = np.arange(len(tr_label))
    np.random.shuffle(index)
    tr_data = tr_data[index, :, :]
    tr_label = tr_label[index]
    
    # randomly select classes
    class_index = np.random.permutation(10)[0:n_way]
    
    # select samples
    sel_tr_data = []
    sel_tr_label = []
    sel_te_data = []
    sel_te_label = []
    for ind, lab in enumerate(class_index):
        # select training samples
        sel_index = np.where(tr_label == lab)[0]
        sel_tr_data.append(tr_data[sel_index[0:n_shot], :, :])
        sel_tr_label.append(np.ones(n_shot, dtype=int)*ind)
        
        # select test samples
        sel_index = np.where(te_label == lab)[0]
        sel_te_data.append(te_data[sel_index, :,:])
        sel_te_label.append(np.ones(len(sel_index), dtype=int)*ind)
    
    sel_tr_data = np.concatenate(sel_tr_data, 0)
    sel_tr_label = np.concatenate(sel_tr_label, 0)
    sel_te_data = np.concatenate(sel_te_data, 0)
    sel_te_label = np.concatenate(sel_te_label, 0)
    # disorder dataset
    index = np.arange(len(sel_tr_label))
    np.random.shuffle(index)
    sel_tr_data = sel_tr_data[index, :, :]
    sel_tr_label = sel_tr_label[index]

    return sel_tr_data, sel_tr_label, sel_te_data, sel_te_label


# depth concat getting data function------------------------------------------------------------------------------------------
# combine training datasets
def com_train_data(tr_label, sam_num, com_label_str):
    if com_label_str == 'all':
        ind_list  = np.arange(len(tr_label))
        str_code = 'np.array(np.meshgrid('
        for j in range(sam_num):
            str_code = str_code + 'ind_list,'
        str_code = str_code + ')).T.reshape(-1, sam_num)'
        index_arr = eval(str_code)
        
    elif com_label_str == 'cat':
        index_arr = []
        for i in range(np.max(tr_label) + 1):
            # find sample index with specific label
            ind_list = np.where(tr_label == i)[0]
            str_code = 'np.array(np.meshgrid('
            for j in range(sam_num):
                str_code = str_code + 'ind_list,'
            str_code = str_code + ')).T.reshape(-1, sam_num)'
            index_arr.append(eval(str_code))
        index_arr = np.concatenate(index_arr, 0)
    # disorder dataset
    index = np.arange(len(index_arr))
    np.random.shuffle(index)
    index_arr = index_arr[index, :] 
    
    return index_arr    

# get training batch data
def get_train_data_depth_concat(ind, bat_num, sam_num, n_way, com_label_str, com_img_str, tr_index, tr_data, tr_label):
    # get the batch index
    if bat_num >= tr_index.shape[0]:
        bat_index = tr_index
        ind = 0
    else:
        st_num = (ind*bat_num)%len(tr_index)
        bat_index = tr_index[st_num: st_num + bat_num, :]
        if bat_index.shape[0] < bat_num:
            ind = 0
        else:
            ind = ind + 1
    len_bat = bat_index.shape[0]
    wid = 28
    
    # get the combine labels
    if com_img_str == 'depth':
        tr_bat_in = np.zeros([len_bat, wid, wid, sam_num])
        for j in range(sam_num):
            tr_bat_in[:, :, :, j] = tr_data[bat_index[:, j], :, :] 
                
    elif com_img_str == 'width':
        tr_bat_in = np.zeros([len_bat, wid, wid*sam_num, 1])
        for j in range(sam_num):
            tr_bat_in[:, :, j*wid: (j+1)*wid, 0] = tr_data[bat_index[:, j], :, :]

    elif com_img_str == 'width_gap':
        tr_bat_in = np.zeros([len_bat, wid, wid*sam_num + 3*(sam_num -1), 1])
        tr_bat_in[:, :, 0:wid, 0] = tr_data[bat_index[:, 0], :, :]
        for j in range(1, sam_num):
            tr_bat_in[:, :, j*wid+3: (j+1)*wid + 3, 0] = tr_data[bat_index[:, j], :, :]
        
    # get the combine label
    if com_label_str == 'all':
        str_code = 'np.zeros([len_bat'
        for i in range(sam_num):
            str_code = str_code + ',n_way'
        str_code = str_code + '])'
        tr_bat_out =  eval(str_code)
        
        for i in range(len_bat):
            str_code = 'tr_bat_out[' + str(i)
            for j in range(sam_num):
                str_code = str_code + ',' + str(tr_label[bat_index[i, j]])
            str_code = str_code + '] = 1'
            exec(str_code)
        
        tr_bat_out = np.reshape(tr_bat_out, [len_bat, n_way**sam_num])            
                
    elif com_label_str == 'cat':
        tr_bat_out = np.zeros([len_bat, n_way])
        for i in range(len_bat):
            tr_bat_out[i, tr_label[bat_index[i, 0]]] = 1
    return tr_bat_in, tr_bat_out, ind

# get traditional training data
def get_train_data_traditional(ind, bat_num, sam_num, n_way, com_label_str, com_img_str, tr_index, tr_data, tr_label):
        # get the batch index
    if bat_num >= tr_index.shape[0]:
        bat_index = tr_index
        ind = 0
    else:
        st_num = (ind*bat_num)%len(tr_index)
        bat_index = tr_index[st_num: st_num + bat_num, :]
        if bat_index.shape[0] < bat_num:
            ind = 0
        else:
            ind = ind + 1
    len_bat = bat_index.shape[0]
    wid = 28
    
    # get the combine labels
    if com_img_str == 'depth':
        tr_bat_in = np.zeros([len_bat, wid, wid, sam_num])
        for j in range(sam_num):
            tr_bat_in[:, :, :, j] = tr_data[bat_index, :, :]
                
    elif com_img_str == 'width':
        tr_bat_in = np.zeros([len_bat, wid, wid*sam_num, 1])
        for j in range(sam_num):
            tr_bat_in[:, :, j*wid: (j+1)*wid, 0] = tr_data[bat_index, :, :]

    elif com_img_str == 'width_gap':
        tr_bat_in = np.zeros([len_bat, wid, wid*sam_num + 3*(sam_num -1), 1])
        tr_bat_in[:, :, 0:wid, 0] = tr_data[bat_index, :, :]
        for j in range(1, sam_num):
            tr_bat_in[:, :, j*wid+3: (j+1)*wid + 3, 0] = tr_data[bat_index, :, :]
        
    # get the combine label
    if com_label_str == 'all':
        str_code = 'np.zeros([len_bat'
        for i in range(sam_num):
            str_code = str_code + ',n_way'
        str_code = str_code + '])'
        tr_bat_out =  eval(str_code)
        
        for i in range(len_bat):
            str_code = 'tr_bat_out[' + str(i)
            for j in range(sam_num):
                str_code = str_code + ',' + str(tr_label[bat_index[i, j]])
            str_code = str_code + '] = 1'
            exec(str_code)
        
        tr_bat_out = np.reshape(tr_bat_out, [len_bat, n_way**sam_num])            
                
    elif com_label_str == 'cat':
        tr_bat_out = np.zeros([len_bat, n_way])
        for i in range(len_bat):
            tr_bat_out[i, tr_label[bat_index[i, 0]]] = 1
    return tr_bat_in, tr_bat_out, ind
    

# get the test batch data
def get_test_data_depth_concat(sam_num, n_way, com_label_str, com_img_str, te_data, te_label):
    len_bat = len(te_label)
    wid = 28
    # get the combine labels
    if com_img_str == 'depth':
        te_bat_in = np.zeros([len_bat, wid, wid, sam_num])
        for j in range(sam_num):
            te_bat_in[:, :, :, j] = te_data
                
    elif com_img_str == 'width':
        te_bat_in = np.zeros([len_bat, wid, wid*sam_num, 1])
        for j in range(sam_num):
            te_bat_in[:, :, j*wid: (j+1)*wid, 0] = te_data

    elif com_img_str == 'width_gap':
        te_bat_in = np.zeros([len_bat, wid, wid*sam_num + 3*(sam_num -1), 1])
        te_bat_in[:, :, 0:wid, 0] = te_data
        for j in range(1, sam_num):
            te_bat_in[:, :, j*wid+3: (j+1)*wid + 3, 0] = te_data
    
    # get the combine label
    if com_label_str == 'all':           
        str_code = 'np.zeros([len_bat'
        for i in range(sam_num):
            str_code = str_code + ',n_way'
        str_code = str_code + '])'
        te_bat_out =  eval(str_code)

        for i in range(len_bat):
            str_code = 'te_bat_out[' + str(i)
            for j in range(sam_num):
                str_code = str_code + ',' + str(te_label[i])
            str_code = str_code + '] = 1'
            exec(str_code)
        
        te_bat_out = np.reshape(te_bat_out, [len_bat, n_way**sam_num]) 
            
    elif com_label_str == 'cat':
        te_bat_out = np.zeros([len_bat, n_way])
        for i in range(len_bat):
            te_bat_out[i, te_label[i]] = 1 
        
    return te_bat_in, te_bat_out

# This function is used to get the global loss
def get_global_loss(tr_loss, tr_index):
    loss_arr = np.stack(tr_loss, 0)
    beg_index = 0
    beg_sum = 0
    gl_loss = []
    for i in range(loss_arr.shape[0]):
        sum_st = np.sum(loss_arr[0:i+1, 3])
        if sum_st - beg_sum == tr_index.shape[0]:
            sum_loss = np.sum(loss_arr[beg_index:i+1, 1]*loss_arr[beg_index:i+1, 3])
            gl_loss.append(sum_loss)
            # update
            beg_index = i+1
            beg_sum = sum_st
            
    return gl_loss
            
        
# analysis the result
def anal_res(te_label, predict, sam_num, n_way):
    str_code = 'np.reshape(predict, [len(te_label)'
    for i in range(sam_num):
        str_code = str_code + ', n_way'
    str_code = str_code + '])'
    prob = eval(str_code)
    sum_res = 0
    for i in range(sam_num):
        sum_num = np.sum(prob, axis = tuple(np.delete(np.arange(1, sam_num+1), i)))
        sum_res = sum_res + sum_num
    # sum
    prob_label = np.argmax(sum_res, axis = 1)
    diff = te_label - prob_label
    
    acc = float(len(np.where(diff == 0)[0]))/len(te_label)
    
    return acc


# combine multiple strings to a single long string
def com_mul_str(str_li):
    str_li = str_li + [''.join(random.sample(string.ascii_letters + string.digits, 4))]
    long_str = ''
    
    for s in str_li:
        long_str = long_str + '_' + str(s)

    return long_str[1: ]
    
# save the model result
def save_result(result, strs):
    filename = '/project/huangli/Project/Project5/MNIST_model/model11_comb_way_gpu/result/result_' + strs + '.txt'
    with open(filename, 'wb') as f1:
        pickle.dump(result, f1)
        
    return

# read the model result
def read_result(filename):
    # get the data of the result
    file_fullname = './result/' + filename
    with open(file_fullname, 'rb') as f1:
        data = pickle.load(f1)
    
    # get the configuration of the result
    sp_str = filename[0:-4].split('_')
    if len(sp_str) == 8:
        conf = [int(sp_str[1]), int(sp_str[2]), int(sp_str[3]), sp_str[4], sp_str[5], sp_str[6]]
    else:
        conf = [int(sp_str[1]), int(sp_str[2]), int(sp_str[3]), sp_str[4], sp_str[5] + '_' + sp_str[6], sp_str[7]]
    
    return data, conf








