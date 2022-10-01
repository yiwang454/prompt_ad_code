import pandas as pd
import sys
import os
import bisect
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support
import re
import random
#prepare to compute n best 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def post_process_bigcross(valid_sp_dict, mode, is_cv=True):
    all_train_df = pd.read_csv('/project_bdda5/bdda/ywang/class_ncd/data/latest_tmp_dir/' + 'train_chas_A.csv')
    all_test_df = pd.read_csv('/project_bdda5/bdda/ywang/class_ncd/data/latest_tmp_dir/' + 'test_chas_A.csv')

    if is_cv:
        train_label = all_train_df.ad.values
    else:
        test_label = all_test_df.ad.values
        # print(test_label)

    if is_cv:
        valid_sp_list = all_train_df.id.values
    else:
        valid_sp_list = all_test_df.id.values

    if mode == 'm_vote':
        preds_sum = []
    
        for v_sp in valid_sp_list:
            pre_arr = np.array(valid_sp_dict[v_sp])
            predict = np.sum(pre_arr)
            preds_sum.append(predict)

        half = pre_arr.size // 2


        preds_sum = np.array(preds_sum)
        preds_new = np.zeros(preds_sum.shape)
        preds_new[preds_sum > half] = 1
        if is_cv:
            metrics = [accuracy_score(train_label, preds_new), precision_score(train_label, preds_new), recall_score(train_label, preds_new), f1_score(train_label, preds_new)]
        else:
            metrics = [accuracy_score(test_label, preds_new), precision_score(test_label, preds_new), recall_score(test_label, preds_new), f1_score(test_label, preds_new)]
        if is_cv:
            wrong_spk_idx = np.where(np.logical_xor(train_label, preds_new))[0]
        else:
            wrong_spk_idx = np.where(np.logical_xor(test_label, preds_new))[0]
            

    return metrics, wrong_spk_idx, preds_new


if __name__ == "__main__": 
    # Hyper parameters shared by both
    merge_way = sys.argv[1]
    MODE = 'm_vote'
    valid_sp_len_list = []
    np.random.seed(42)


    #hyper parameters for not fixed epoch only
    accuracies_list = []
    combo_list = []

    if merge_way == 'rand_merge_cv':
        ckpt_dir = sys.argv[2] #"/project_bdda7/bdda/ywang/Public_Clinical_Prompt/logs/bert-base-uncased_tempmanual0_verbmanual_full_100/version_85_val"

        list_acc = []
        cls_app = 'svm'
        valid_sp_dict = {}
        valid_label_dict = {}
        for i in [1]:
            for j in range(10):
                all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results_cv{}_fold{}.csv'.format(i, j)))

                for row in range(len(all_result_df)):
                    valid_sp_dict[all_result_df['id'][row]] = all_result_df['pred_labels'][row]
                    valid_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

            # for v_sp in valid_sp_dict:
            #     assert len(valid_sp_dict[v_sp]) == 10
            cross_out0, _, _ = post_process_bigcross(valid_sp_dict, mode=MODE, is_cv=True)
            f_out_str = ''
            for m in cross_out0:
                f_out_str += '{:.4f} & '.format(m)
            
            print(f_out_str)
    elif merge_way == 'rand_test':
        ckpt_root = sys.argv[2]
        assert 'val' not in ckpt_root
        list_acc = []
        cls_app = 'svm'

        for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            
            ckpt_dir = ckpt_root.rstrip('_') + '_{}'.format(seed)
            test_sp_dict = {}
            test_label_dict = {}

            all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results.csv'))

            for row in range(len(all_result_df)):
                test_sp_dict[all_result_df['id'][row]] = all_result_df['pred_labels'][row]
                test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

            cross_out0, _, _ = post_process_bigcross(test_sp_dict, mode=MODE, is_cv=False)

            print(seed, '{:.4f}'.format(cross_out0[0]))
    else:
        NotImplemented