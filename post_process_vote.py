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



    if merge_way == 'rand_test':
        ckpt_root = sys.argv[2]
        assert 'val' not in ckpt_root
        list_acc = []
        cls_app = 'svm'
        all_wrong_speakerlist = []
        for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            
            ckpt_dir = ckpt_root.rstrip('/') + '/version_{}'.format(seed)
            test_sp_dict = {}
            test_label_dict = {}

            all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results_last.csv'))

            for row in range(len(all_result_df)):
                test_sp_dict[all_result_df['id'][row]] = all_result_df['pred_labels'][row]
                test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

            cross_out0, wrong_spk_idx, _ = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)

            print(seed, '{:.4f}'.format(cross_out0[0]))
            all_wrong_speakerlist.extend(test_sp_list[wrong_spk_idx])
        
        # count_wrong_spk = Counter(all_wrong_speakerlist)
        # with open(os.path.join(ckpt_root, 'wrong_spk.json'), 'w+') as json_w:
        #     json.dump(dict(count_wrong_spk), json_w)
        
    elif merge_way == 'rand_test_emg': #  epoch merge
        ckpt_root = sys.argv[2]
        
        list_acc = []
        template_id = [3, 1]
        with open(os.path.join('./prompt_ad_code/latest_tmp_dir', 'test_all_spk.json'), 'r') as j_read:
            bert_list_speakers = json.load(j_read)
        
        n_best_idx = [7, 8, 9]

        for tem_id in template_id:
            print('tem_id', tem_id)
            for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
                test_sp_dict = {s: [] for s in test_sp_list}
                ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}'.format(seed))
                
                # test_label_dict = {}
                
                for idxes in n_best_idx:
                    all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'test_results_epoch{}.csv'.format(idxes)))

                    for row in range(len(all_result_df)):
                        test_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                        # test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]
                
                latest_cross_out0, _, pre_new = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)

                print(seed, '{:.4f}'.format(latest_cross_out0[0]))
    
    elif merge_way == 'rand_test_merge':
        ckpt_root = sys.argv[2]
        
        list_acc = []
        cls_app = 'svm'
        template_id = []
        with open(os.path.join('./prompt_ad_code/latest_tmp_dir', 'test_all_spk.json'), 'r') as j_read:
            bert_list_speakers = json.load(j_read)

        n_best_idx = [7, 8, 9]
        plot_acc_list = []
        right_tie_speakerlist = []
        wrong_tie_speakerlist = []
        for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            test_sp_dict = {s: [] for s in test_sp_list}
            for tem_id in template_id:
                ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}'.format(seed))
                # -uncased
                # test_label_dict = {}

                for idxes in n_best_idx:
                    all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'epoch{}'.format(idxes), 'test_results.csv'))

                    for row in range(len(all_result_df)):
                        test_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                        # test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]
            
            latest_cross_out0, _, pre_new = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)
            print(seed, '{:.4f}'.format(latest_cross_out0[0]))
            plot_acc_list.append(latest_cross_out0[0])

            for k, t_sp in enumerate(bert_list_speakers):
                if test_sp_dict[t_sp].count(1) == len(test_sp_dict[t_sp]) // 2:
                    if pre_new[k] == int(test_label[k]):
                        right_tie_speakerlist.append(t_sp)
                    else:
                        wrong_tie_speakerlist.append(t_sp)
                
        
        print(len(test_sp_dict['S191']))
        
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print('{:.4f}'.format(combo_avg))
        combo_std = np.std(combo_arr, axis=0)
        print('{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))

    elif merge_way == 'rand_test_robbertmg':
        ckpt_root = sys.argv[2]
        
        list_acc = []
        template_id = [1, 3]
        with open(os.path.join('./prompt_ad_code/latest_tmp_dir', 'test_all_spk.json'), 'r') as j_read:
            bert_list_speakers = json.load(j_read)

        n_best_idx = [7, 8, 9]
        plot_acc_list = []
        right_tie_speakerlist = []
        wrong_tie_speakerlist = []
        for bert_seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 
            for roberta_seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]: # 

                test_sp_dict = {s: [] for s in test_sp_list}
                for tem_id in template_id:
                    ckpt_dir = os.path.join(ckpt_root, 'bert-base-uncased_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}'.format(bert_seed))
                    ckpt_dir_roberta = os.path.join(ckpt_root, 'roberta-base_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}'.format(roberta_seed))
                    # test_label_dict = {}

                    for idxes in n_best_idx:
                        all_result_df = pd.read_csv(os.path.join(ckpt_dir, 'checkpoints', 'epoch{}'.format(idxes), 'test_results.csv'))
                        all_result_df_roberta = pd.read_csv(os.path.join(ckpt_dir_roberta, 'checkpoints', 'epoch{}'.format(idxes), 'test_results.csv'))

                        for row in range(len(all_result_df)):
                            test_sp_dict[all_result_df['id'][row]].append(all_result_df['pred_labels'][row])
                            test_sp_dict[all_result_df_roberta['id'][row]].append(all_result_df_roberta['pred_labels'][row])
                            # test_label_dict[all_result_df['id'][row]] = all_result_df['labels'][row]

                
                latest_cross_out0, _, pre_new = post_process_bigcross(test_sp_dict, test_sp_list, test_label, mode=MODE)
                # print(seed, '{:.4f}'.format(latest_cross_out0[0]))
                plot_acc_list.append(latest_cross_out0[0])
                for k, t_sp in enumerate(bert_list_speakers):
                    if test_sp_dict[t_sp].count(1) == len(test_sp_dict[t_sp]) // 2:
                        if pre_new[k] == int(test_label[k]):
                            right_tie_speakerlist.append(t_sp)
                        else:
                            wrong_tie_speakerlist.append(t_sp)
        
        print(len(test_sp_dict['S191']))
        print(len(plot_acc_list))
        
        combo_arr = np.array(plot_acc_list)
        combo_avg = np.mean(combo_arr, axis=0)
        print('{:.4f}'.format(combo_avg))
        combo_std = np.std(combo_arr, axis=0)
        print('{:.4f}'.format(combo_std))
        combo_max = np.max(combo_arr, axis=0)
        print('{:.4f}'.format(combo_max))


    else:
        NotImplemented