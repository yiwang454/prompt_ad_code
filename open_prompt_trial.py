from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.plms import T5TokenizerWrapper
from openprompt import Template
from openprompt import PromptDataLoader


from transformers.tokenization_utils import PreTrainedTokenizer

from prompt_ad_utils import read_input_text
import argparse
import pandas as pd
import torch

# class ManualTemplate_mod(Template):
#     """
#     Args:
#         tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
#         text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
#         placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
#     """

#     registered_inputflag_names = ["loss_ids", 'shortenable_ids']

#     def __init__(self,
#                  tokenizer: PreTrainedTokenizer,
#                  text = None,
#                  placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'}, # ,'<text_c>':'text_c','<ans_a>':'ans_a','<ans_b>':'ans_b'
#                 ):
#         super().__init__(tokenizer=tokenizer,
#                          placeholder_mapping=placeholder_mapping)
#         self.text = text

#     def on_text_set(self):
#         self.text = self.parse_text(self.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="bert-base-uncased")
    parser.add_argument("--data_dir", type=str, default="/project_bdda5/bdda/ywang/class_ncd/data/latest_tmp_dir/")

    args = parser.parse_args()

    data_save_dir = args.data_dir
    bert_model_dir = args.model_dir
    

    MANUAL_TYPE = 'A'
    SAMPLE_SIZE = 3
    classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
        "healthy",
        "dementia"
    ]

    if torch.cuda.is_available():
        # use_cuda = True
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda')
    else:
        # use_cuda = False
        device = torch.device('cpu')

    train_df = pd.read_csv(data_save_dir + 'train_chas_{:s}.csv'.format(MANUAL_TYPE)) # transcripts
    test_df = pd.read_csv(data_save_dir + 'test_chas_{:s}.csv'.format(MANUAL_TYPE))

    train_data_df = read_input_text(train_df, sample_size=3)
    test_data_df = read_input_text(test_df, sample_size=3)

    dataset = {'train': [], 'test': []}
    if SAMPLE_SIZE == -1:
        for index, data in train_data_df.iterrows():
            input_example = InputExample(text_a = data['joined_all_par_trans'], label=data['ad'], guid=index)
            dataset['train'].append(input_example)

        for index, data in test_data_df.iterrows():
            input_example = InputExample(text_a = data['joined_all_par_trans'], label=data['ad'], guid=index)
            dataset['test'].append(input_example)
    elif SAMPLE_SIZE > 0:
        for idx, trn_df in enumerate(train_data_df):
            meta = {
            "text_c" : trn_df.joined_all_par_trans[1],
            "text_d" : trn_df.joined_all_par_trans[2],
            "ans_c" : classes[trn_df.ad[1]],
            "ans_d" : classes[trn_df.ad[2]],}
            input_example = InputExample(text_a = trn_df.joined_all_par_trans[0], 
            label=trn_df.ad[0], meta=meta, guid=idx)
            dataset['train'].append(input_example)

        for idx, trn_df in enumerate(test_data_df):
            meta = {
            "text_c" : trn_df.joined_all_par_trans[1],
            "text_d" : trn_df.joined_all_par_trans[2],
            "ans_c" : classes[trn_df.ad[1]],
            "ans_d" : classes[trn_df.ad[2]],}
            input_example = InputExample(text_a = trn_df.joined_all_par_trans[0], 
            label=trn_df.ad[0], meta=meta, guid=idx)
            dataset['test'].append(input_example)
    else:
        raise NotImplementedError

    # for split in ['train', 'test']:
    #     dataset[split] = [ # For simplicity, there's only two examples
    #     # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    #     InputExample(
    #         guid = 0,
    #         text_a = "Albert Einstein was one of the greatest intellects of his time.",
    #     ),
    #     InputExample(
    #         guid = 1,
    #         text_a = "The film was badly made.",
    #     ),
    # ]
        # for data in raw_dataset[split]:
        #     input_example = InputExample(text_a = data['premise'], text_b = data['hypothesis'], label=int(data['label']), guid=data['idx'])
        #     dataset[split].append(input_example)
    # print(dataset['train'][0])


    # bert_model_dir = '/project_bdda5/bdda/ywang/class_ncd/new_models/bert-base-uncased'
    # bert_model_dir = '/project_bdda5/bdda/ywang/class_ncd/new_models/bert_post_train_total_loss_lr00001_02_1/step1653'

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", bert_model_dir)
    promptTemplate = ManualTemplate(
        text = '{"meta": "text_c"}. Diagnosis is {"meta": "ans_c"}. {"meta": "text_d"}. Diagnosis is {"meta": "ans_d"}. {"placeholder": "text_a"}. Diagnosis is {"mask"}.', #'{"placeholder":"text_a"} Diagnosis is {"mask"}', 
        tokenizer = tokenizer,
    )

    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "dementia": ["dementia", "disorder", "disease", "declined"],
            "healthy": ["healthy", "normal", "fit"],
        },
        tokenizer = tokenizer,
    )
    # print(promptVerbalizer.label_words_ids)


    promptModel = PromptForClassification(
        template = promptTemplate,
        plm = plm,
        verbalizer = promptVerbalizer,
    ).to(device)

    wrapped_example = promptTemplate.wrap_one_example(dataset['train'][0])
    print(wrapped_example)

    # wrapped_t5tokenizer= T5TokenizerWrapper(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")
    # tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)




    # logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm, and
    # print(promptVerbalizer.process_logits(logits)) # see what the verbalizer do

    train_data_loader = PromptDataLoader(
        dataset = dataset['train'],
        tokenizer = tokenizer, 
        template = promptTemplate, 
        tokenizer_wrapper_class=WrapperClass,
    )

    test_data_loader = PromptDataLoader(
        dataset = dataset['test'],
        tokenizer = tokenizer, 
        template = promptTemplate, 
        tokenizer_wrapper_class=WrapperClass,
    )


    promptModel.eval()

    def evaluation(train_data_loader, device):
        with torch.no_grad():
            correct = 0
            all_size = len(train_data_loader)
            AD_size = 0
            for batch in train_data_loader:
                AD_size += batch.label.cpu().numpy()
                batch = batch.to(device)
                logits = promptModel(batch)
                preds = torch.argmax(logits, dim = -1)
                correct += preds.eq(batch.label)
            print(correct/all_size)
            print(AD_size/all_size)

    evaluation(train_data_loader, device)
    evaluation(test_data_loader, device)
