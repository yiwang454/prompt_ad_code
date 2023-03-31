import subprocess
import os
import sys
CWD = sys.argv[1] # Input your working directory

GPU_idx = 0

COMMAND_LIST = []
for template_id in [1, 3]:
    for seed in [1, 2, 10, 18, 26, 31, 32, 52, 61, 68, 70, 72, 85, 93, 94]:
        for fold_idx in range(10):
            command = '''python prompt_ad_code/prompt_finetune.py \
                    --project_root /parent/directory/of_prompt_ad_code \
                    --logs_root /directory/to/store/your/output \
                    --off_line_model_dir /directory/you/store/pre-trained/model/from/huggingface \
                    --data_dir /directory/you/store/ADReSS/data \
                    --data_not_saved \
                    --seed {:d} \
                    --tune_plm \
                    --model bert \
                    --model_name bert-base-uncased \
                    --template_type manual --verbalizer_type manual \
                    --template_id {:d} \
                    --gpu_num {} \
                    --num_epochs 10 \
                    --no_ckpt True \
                    --last_ckpt \
                    --crossvalidation \
                    --val_file_dir prompt_ad_code/latest_tmp_dir/ten_fold_1.json \
                    --val_fold_idx {} \
                    --no_tensorboard '''.format(seed, template_id, GPU_idx, fold_idx)
            COMMAND_LIST.append(command)

for k in range(len(COMMAND_LIST)):
    subp = subprocess.Popen(COMMAND_LIST[k], shell=True, cwd=CWD, encoding="utf-8")
    subp.wait()
    
    if subp.poll() == 0:
        print(subp.communicate())
    else:
        print(COMMAND_LIST[k], 'fail')
        with open('./running_status.txt', 'a+') as run_write:
            run_write.write(COMMAND_LIST[k] + '  fail\n')