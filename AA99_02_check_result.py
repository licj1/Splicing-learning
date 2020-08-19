# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:20:58 2019

@author: pc
"""
# This script is used to read the model saved result

import numpy as np
import support_based as spb
import os


pa_dir = os.listdir('./result/')

res_li = []
for f in pa_dir:
    data, conf = spb.read_result(f)
    acc_max = np.max(data[2][:, 3])
    res_li.append(np.array(conf + [acc_max]))
    
# convert list to array
res_arr = np.stack(res_li, 0)


sati_res_li = []
for n_way in [5]:
    for n_shot in [5]:
        for sam_num in [1, 2, 3, 4, 5]:
            for com_label_str in ['all', 'cat']:
                for com_img_str in ['depth', 'width', 'width_gap']:
                    for com_tradi in ['com', 'tradi']:
                        # selecting
                        sel_res_arr = res_arr[res_arr[:, 0] == str(n_way), :]
                        sel_res_arr = sel_res_arr[sel_res_arr[:, 1] == str(n_shot)]
                        sel_res_arr = sel_res_arr[sel_res_arr[:, 2] == str(sam_num)]
                        sel_res_arr = sel_res_arr[sel_res_arr[:, 3] == com_label_str]
                        sel_res_arr = sel_res_arr[sel_res_arr[:, 4] == com_img_str]
                        sel_res_arr = sel_res_arr[sel_res_arr[:, 5] == com_tradi]
                        sati_res_li.append(np.array([n_way, n_shot, sam_num, com_label_str, com_img_str, com_tradi, sel_res_arr.shape[0], np.mean(sel_res_arr[:, -1].astype(float)), np.std(sel_res_arr[:, -1].astype(float))]))
                        
                        
sati_res_li = np.stack(sati_res_li, 0)
                        
                        
                        


