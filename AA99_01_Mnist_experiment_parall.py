# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:47:33 2019

@author: pc
"""
import support_based as spb
import CNN_support as cp
import CNN_support_traditional as cpt

# load data
tr_image, tr_label, te_image, te_label = spb.load_data()

for times in range(1):
    for n_way in [5]:
        for n_shot in [5]:
            once = False
            sel_tr_img, sel_tr_label , sel_te_img, sel_te_abel = spb.select_samples(tr_image, tr_label, te_image, te_label, n_way, n_shot)
            for sam_num in [5]:
                for com_label_str in ['all', 'cat']:
                    for com_img_str in ['depth', 'width', 'width_gap']:
                        
                        if sam_num == 1 and once == True:
                            continue
                        else:
                            once = True
                            str1 = spb.com_mul_str([n_way, n_shot, sam_num, com_label_str, com_img_str, 'com'])
                            cp.CNNnet(sel_tr_img, sel_tr_label, sel_te_img, sel_te_abel, sam_num, n_way, com_label_str, com_img_str, str1, iter_num = 200000)
                            if sam_num > 1:
                                str2 = spb.com_mul_str([n_way, n_shot, sam_num, com_label_str, com_img_str, 'tradi'])
                                cpt.CNNnet(sel_tr_img, sel_tr_label, sel_te_img, sel_te_abel, sam_num, n_way, com_label_str, com_img_str, str2, iter_num = 200000)




