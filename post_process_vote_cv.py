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
from collections import Counter
import matplotlib.pyplot as plt


def get_validation_idx(current_model_dir):
    # current model dir example: new_models/bert_post_train_total_loss_lr000005_1_cv3_fold2
    data_save_dir = './prompt_ad_code/latest_tmp_dir'
    model_dir_tails = current_model_dir.split('lr')[-1]
    validation_info = model_dir_tails.split('_')[2:]
    if 'cv' in validation_info[0]:
        #validation_mode = cv
        validation_idx_file = os.path.join(data_save_dir, 'ten_fold_{}.json'.format(validation_info[0].lstrip('cv')))
        fold_idx = int(validation_info[-1].lstrip('fold'))
    else:
        raise ValueError('validation info wrong')

    with open(validation_idx_file, 'r') as v_read:
        validation_dict = json.load(v_read)
    
    if fold_idx in [idx for idx in range(0, 10)]:
        train_speaker = validation_dict[fold_idx]['train_speaker']
        validation_speaker = validation_dict[fold_idx]['test_speaker']
        return train_speaker, validation_speaker
    else:
        raise ValueError('validation index wrong')

def plot_wrong_spk(wrong_spk_dict, file_name):
    
    plt.figure()
    plt.bar(wrong_spk_dict.keys(), wrong_spk_dict.values(), color='lightgreen',
                            alpha=0.7, width=0.85)

    plt.xlabel('speaker ID')
    plt.ylabel('Error detect Frequency')
    plt.xticks(rotation=90)
    # plt.legend()
    plt.savefig(file_name)

def post_process_bigcross(valid_sp_dict, valid_sp_list, valid_label, mode):

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
        metrics = [accuracy_score(valid_label, preds_new), precision_score(valid_label, preds_new), recall_score(valid_label, preds_new), f1_score(valid_label, preds_new)]

        wrong_spk_idx = np.where(np.logical_xor(valid_label, preds_new))[0]

    return metrics, wrong_spk_idx, preds_new

def str_to_int(str):
    return list(map(int, str.strip('[').strip(']').split(' ')))


if __name__ == "__main__": 
    # Hyper parameters shared by both
    merge_way = sys.argv[1]
    MODE = 'm_vote'
    valid_sp_len_list = []
    np.random.seed(42)


    #hyper parameters for not fixed epoch only
    accuracies_list = []
    combo_list = []

    all_train_df = pd.read_csv('./prompt_ad_code/latest_tmp_dir' + 'train_chas_A.csv')
    all_test_df = pd.read_csv('./prompt_ad_code/latest_tmp_dir' + 'test_chas_A.csv')

    train_label = all_train_df.ad.values
    test_label = all_test_df.ad.values
    # print(test_label)

    train_sp_list = all_train_df.id.values
    test_sp_list = all_test_df.id.values


    if merge_way == 'rand_cv':
        ckpt_dir = sys.argv[2] #
        list_acc = []
        cls_app = 'svm'
        valid_sp_dict = {}
        valid_label_dict = {}
        for i in [1]:
            for j in range(10):
                all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results_last_cv{}_fold{}.csv'.format(i, j)))

                for row in range(len(all_result_df)):
                    valid_sp_dict[all_result_df['id'][row]] = all_result_df['pred_labels'][row]
                    valid_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

            # for v_sp in valid_sp_dict:
            #     assert len(valid_sp_dict[v_sp]) == 10
            cross_out0, _, _ = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
            
            f_out_str = ''
            for m in cross_out0:
                f_out_str += '{:.4f} & '.format(m)
            
            print(f_out_str)


    elif merge_way == 'rand_cv_emg':
        ckpt_root = sys.argv[2]

        list_acc = []
        template_id = [1, 3]
        n_best_idx = [7, 8, 9]
        plot_acc_list = []

        for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            valid_sp_dict = {s: [] for s in train_sp_list}
            for tem_id in template_id:
                ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_full_lr100'.format(tem_id), 'version_{}_val'.format(seed))

                # for i in [1]:
                for idxes in n_best_idx:
                    for j in range(10):
                        file_path = os.path.join(ckpt_dir, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j))
                        if not os.path.exists(file_path):
                            print(file_path)
                            continue
                        all_result_df = pd.read_csv(file_path)
                        for row in range(len(all_result_df)):
                            valid_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])

            # for v_sp in valid_sp_dict:
            #     assert len(valid_sp_dict[v_sp]) == 10
            cross_out0, _, _ = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
            # print(valid_sp_dict['S083'])
            print(seed, '{:.4f}'.format(cross_out0[0]))
            plot_acc_list.append(cross_out0[0])
        
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print('{:.4f}'.format(combo_avg))
        combo_std = np.std(combo_arr, axis=0)
        print('{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))

    elif merge_way == 'rand_cv_merge':
        ckpt_root = sys.argv[2]
        
        list_acc = []
        cls_app = 'svm'
        template_id = [1, 3]

        n_best_idx = [7, 8, 9]
        # backbone = sys.argv[3]
        plot_acc_list = []

        for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            valid_sp_dict = {s: [] for s in train_sp_list}
            for tem_id in template_id:
                ckpt_dir = os.path.join(ckpt_root, 'roberta-base_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}_val'.format(seed))
                # bert-base-uncased roberta-base
                # test_label_dict = {}

                for idxes in n_best_idx:
                    for j in range(10):
                        file_path = os.path.join(ckpt_dir, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j))
                        all_result_df = pd.read_csv(file_path)

                        for row in range(len(all_result_df)):
                            valid_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
            
            
            cross_out0, _, pre_new = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
            print(seed, '{:.4f}'.format(cross_out0[0]))
            plot_acc_list.append(cross_out0[0])
        
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print(combo_avg)
        print(combo_avg * 225 * 108)
        combo_std = np.std(combo_arr, axis=0)
        print('{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))
        

    elif merge_way == 'rand_cv_robbertmg':
        ckpt_root = sys.argv[2]
        
        list_acc = []
        template_id = [1, 3]
        n_best_idx = [7, 8, 9]
        plot_acc_list = []

        for bert_seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            for roberta_seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
                valid_sp_dict = {s: [] for s in train_sp_list}
                for tem_id in template_id:
                    ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_full_lr100'.format(tem_id), 'version_{}_val'.format(bert_seed))
                    ckpt_dir_roberta = os.path.join(ckpt_root, 'roberta-base_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}_val'.format(roberta_seed))
                    # test_label_dict = {}

                    for idxes in n_best_idx:
                        for j in range(10):
                            all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j)))
                            all_result_df_roberta = pd.read_csv(os.path.join(ckpt_dir_roberta, 'checkpoints/epoch{}'.format(idxes), 'test_results_cv1_fold{}.csv'.format(j)))
                            for row in range(len(all_result_df)):
                                valid_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                                valid_sp_dict[all_result_df_roberta['id'][row]].append(all_result_df_roberta['pred_labels'][row])
                            # for row in range(len(all_result_df_roberta)):

                cross_out0, _, pre_new = post_process_bigcross(valid_sp_dict, train_sp_list, train_label, mode=MODE)
                # print(seed, '{:.4f}'.format(cross_out0[0]))
                plot_acc_list.append(cross_out0[0])
        
        print(len(plot_acc_list))
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print('{:.4f}'.format(combo_avg))
        # print(combo_avg)
        # print(combo_avg * 225 * 108)
        combo_std = np.std(combo_arr, axis=0)
        print('{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))
        


    else:
        NotImplemented