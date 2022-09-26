import subprocess
import os
CWD = '/project_bdda7/bdda/ywang/Public_Clinical_Prompt'

GPU_nums = [0 for _ in range(1)]
ss_num = len(GPU_nums)
GPU_idx = 0

COMMAND_LIST = []

command = '''python prompt_ad/prompt_finetune.py \
        --tune_plm \
        --model bert \
        --model_name bert-tuned \
        --run_evaluation \
        --template_type manual --verbalizer_type manual \
        --template_id 0 \
        --gpu_num {} \
        --num_epochs 10 \
        >> "/project_bdda7/bdda/ywang/Public_Clinical_Prompt/prompt_ad_log.txt"'''.format(GPU_nums[GPU_idx])
GPU_idx = (GPU_idx + 1) % len(GPU_nums)

COMMAND_LIST.append(command)

# can change --optimizer to adamw later
# --ce_class_weights currently not using

for k in range(len(COMMAND_LIST) // ss_num + 1):
    subp_ss = []
    for q in range(ss_num):
        if ss_num * k + q < len(COMMAND_LIST):
            print(COMMAND_LIST[ss_num * k + q], q)

            subp = subprocess.Popen(COMMAND_LIST[ss_num * k + q], shell=True, cwd=CWD, encoding="utf-8") #stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            
            subp_ss.append(subp)            
    
    for q in range(len(subp_ss)):
        subp_ss[q].wait()
    
    for q in range(len(subp_ss)):
        if subp_ss[q].poll() == 0:
            print(subp_ss[q].communicate())
        else:
            print(COMMAND_LIST[ss_num * k + q], 'fail')
            with open('./running_status_get_svm_result.txt', 'a+') as run_write:
                run_write.write(COMMAND_LIST[ss_num * k + q] + '  fail\n')