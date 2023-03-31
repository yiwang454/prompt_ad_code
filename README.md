## Code for ICASSP23 paper "EXPLOITING PROMPT LEARNING WITH PRE-TRAINED LANGUAGE MODELS FOR ALZHEIMER'S DISEASE DETECTION"

## Overview
**Prompt-learning** is the latest paradigm to adapt pre-trained language models (PLMs) to downstream NLP tasks. We hereby explore its application on Alzheimer's disease detection. Our relevant paper is accepted by ICASSP23 and available [here](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9aXKHIYAAAAJ&citation_for_view=9aXKHIYAAAAJ:UeHWp8X0CEIC)

Currently, only codes for the primary results of prompt-based fine-tuning experiments in the paper are provided. More specifically, codes for system 2-4, 7-9, 12-14 cross-validation and test results using Manual transcripts (columns 5-10) in Table 2 of the paper are now available in this repository. Commands for running other experiments in the paper(disfluency feature based and ASR transcripts based) will be coming soon. Meanwhile, users can adopt those features by changing arguments in the provided commands and scripts accordingly.

## Prompt-based Fine-tuning Command
After downloading this repository and solving environment for [transformers](https://github.com/huggingface/transformers) and [OpenPrompt](https://github.com/thunlp/OpenPrompt), you can run the following commands in the parent directory of ```prompt_ad_code``` directory.

Before running the ```run_prompt_finetune.py``` or ```run_prompt_finetune_test.py``` in the following instruction, you'll have to define the ```project_root```, ```logs_root```, ```off_line_model_dir```, ```data_dir``` configurations in your scripts. These configuration should be set to 1) the parent directory of your prompt_ad_code folder; 2) the directory to store your output (model or results); 3) the directory you store pre-trained model downloaded from huggingface; 4) the directory you store ADReSS data (csv file), respectively.

```python
--project_root /parent/directory/of_prompt_ad_code \
--logs_root /directory/to/store/your/output \
--off_line_model_dir /directory/you/store/pre-trained/model/from/huggingface \
--data_dir /directory/you/store/ADReSS/data \
```
Also, you have to change ```CWD = ""``` in ```run_prompt_finetune.py``` or ```run_prompt_finetune_test.py``` scripts into your working directory.

<!-- ```python
CWD = "/directory/you/are/working/in"
``` -->

### Data Split
We adopt the given train and test split from the ADReSS2020 dataset, with train speaker ids all stored in ```prompt_ad_code/latest_tmp_dir/train_all_spk.json```, and test speaker ids all stored in  ```prompt_ad_code/latest_tmp_dir/test_all_spk.json```. In cross validation experiments, we adopt 10 fold cross validtion, with validation split stored in ```prompt_ad_code/latest_tmp_dir/ten_fold_1.json```. 

```ten_fold_1.json``` is a list with 10 entries. After 108 training speakers being splitted into 10 folds, each fold takes turns to serve as the validation set, leading to 10 train-validation set pairs. 10 entries of the list stored the dictionary of {"train_speaker": list_of_train_speaker, "test_speaker": list_of_test_speaker} representing a corresonding train-validation set pair. 


### Data Format
In your data directory, you'd have to store the ADReSS train data in csv file named ```train_chas_A.csv``` and test data in csv file named ```test_chas_A.csv``` with columns named ```id```, ```age```, ```joined_all_par_trans```, ```ad```, **without the index column for csv**. ```joined_all_par_trans``` should store a single string. The string is constructed by all transcript sentences of that corresponding speaker being joined together. 

An example of ```train_chas_A.csv``` without any actual transcripts is in ```prompt_ad_code/data```

You may want to involve a data pre-processing step which cuts the transcripts of each participant into the window length of 512 tokens (cutting at intervals between full sentences, so the resulted length may be slightly less than 512 tokens). This is done by adding ```--data_not_saved``` config into ```run_prompt_finetune_test.py```; adding ```COMMAND_LIST = COMMAND_LIST[:1]``` to around line 30 in ```run_prompt_finetune_test.py``` and run for once. It will give you ```train_chas_A_cut.csv``` and ```test_chas_A_cut.csv``` files storing the cut data. The ```--data_not_saved``` flag will let the script stop running after saving data without running any experiments. 

In real experiments running, please ensure you remove this config from ```run_prompt_finetune_test.py``` and remove the ```COMMAND_LIST = COMMAND_LIST[:1]``` in ```run_prompt_finetune_test.py```.

If you prefer other data formatting, you can change the data loader part in ```prompt_finetune.py``` lines 127-179, and its corresponding functions in ```prompt_ad_utils.py```, and maybe ```prompt_finetune.py``` lines 272-302 accordingly.

### Cross Validation
To run the prompt-based fine-tuning with BERT as the PLM, and get 5 fold cross validation (CV) results:
```
python prompt_ad_code/run_prompt_finetune.py /parent/directory/of/prompt_ad_code
```

To run the prompt-based fine-tuning with RoBERTa as the PLM (CV), pls change ```model``` and ```model_name``` config in ```run_prompt_finetune.py``` to the following
```python
--model roberta \
--model_name roberta-base \
```

And run
```
python prompt_ad_code/run_prompt_finetune.py /parent/directory/of/prompt_ad_code
```

The cross validation results as mentioned above are run on a ten-fold split specified in ```prompt_ad_code/latest_tmp_dir/ten_fold_1.json```. You can use your own split by change ```val_file_dir``` config in ```run_prompt_finetune.py``` to your own split file using similar format as ```ten_fold_1.json```.
```python
--val_file_dir /your/own/split/file.json \
```

### Test
To run the prompt-based fine-tuning BERT PLM test results 
```
python prompt_ad_code/run_prompt_finetune_test.py /parent/directory/of/prompt_ad_code
```
Similar change can be made as in CV to run the roBERTa experiments.

All the above scripts include running fine-tuning experiment with prompt location at the front or at the back of the input texts.  The following line in ```run_prompt_finetune.py``` speficied the template used. ```template_id``` 1 refers to the front location prompt template, while 3 refers to the back location prompt template.
```python
for template_id in [1, 3]
```


## Fine-tuned Model Majority Vote Combination Command
To get the BERT epoch combination results (systems 4, columns 5-10), with majority voting between the last three epochs during fine-tuning, you can run the following command.

### Cross Validation
```
python prompt_ad_code/post_process_vote_cv.py rand_cv_emg /directory/to/store/your/output
```

### Test
```
python prompt_ad_code/post_process_vote.py rand_test_emg /directory/to/store/your/output
```

In the two commands above, the directory to store your output is the same as ```logs_root``` mentioned before. 

To get the RoBERTa epoch combination results (systems 9, columns 5-10), you can simply change the ```bert-base-uncased``` in line 122  of ```post_process_vote.py``` to
```python
ckpt_dir = os.path.join(ckpt_root, 'roberta-base_tempmanual{}_verbmanual_full_100'.format(tem_id), 'version_{}'.format(seed))
```
The cross validation code modification from BERT to RoBERTa is similar.


## Fine-tuned BERT+RoBRETa Model Combination Command
To get the BERT+RoBRETa feature combination results (systems 14, columns 5-10), you can run the following command:

### Cross Validation
```
python prompt_ad_code/post_process_vote_cv.py rand_cv_robbertmg /directory/to/store/your/output
```

### Test
```
python prompt_ad_code/post_process_vote.py rand_test_robbertmg /directory/to/store/your/output
```

## Using front/back prompt template, or using front + back prompt paradigm combination
In the above sections, by default the scripts is reporting front + back location prompt template combination (majority voting results). If you hope to get single template results, pls modify the corresponding lines in the scripts from:
```python
template_id = [3, 1]
```
to front template id:
```python
template_id = [1]
```
or to back template id:
```python
template_id = [3]
```