import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Tuple

import datasets
from datasets import load_dataset
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    BertModel,
    BertForPreTraining,
    PreTrainedTokenizerBase
)
from transformers.data.data_collator import _torch_collate_batch
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

def save_mutable(model, tokenizer, save_directory):
    # save
    if os.path.isfile(save_directory):
        logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
        return
    os.makedirs(save_directory, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        # 'tokenizer': tokenizer
    }, os.path.join(save_directory , 'pytorch_model.bin'))

def load_model(save_directory):
    checkpoint = torch.load(os.path.join(save_directory , 'pytorch_model.bin')) #, map_location='cpu'
    return checkpoint

def save_checkpoint(save_path, model, tokenizer):

    if save_path == None:
        print('no path, no save')
        return
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    #state_dict = {'model_state_dict': model.state_dict(),
    #              'valid_loss': valid_loss}
    
    #torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    #state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    #model.load_state_dict(state_dict['model_state_dict'])
    model.from_pretrained(load_path)

def save_metrics(save_path, global_steps_list,train_mlm_loss_list, train_nsp_loss_list, train_loss_list,eval_mlm_loss_list,eval_nsp_loss_list,eval_loss_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
              'train_mlm_loss_list':train_mlm_loss_list,
              'train_nsp_loss_list':train_nsp_loss_list,
              'eval_loss_list': eval_loss_list,
              'eval_mlm_loss_list':eval_mlm_loss_list,
              'eval_nsp_loss_list':eval_nsp_loss_list,
              'global_steps_list': global_steps_list,}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')



# class MyDataCollator(DataCollatorForLanguageModeling):

#     def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]): # -> Dict[str, torch.Tensor]
#         # Handle dict or lists with proper padding and conversion to tensor.
#         assert isinstance(examples[0], (dict))
#         if "next_sentence_label" in examples[0].keys():
#             assert "ad" not in examples[0].keys()
#             examples_new = [{k:v for k,v in _.items() if k not in ["id", "joined_all_par_trans", "joined_all_par_trans_index", 
#                                                   "previous_sentence", "previous_sentence_label"]} 
#                        for _ in examples]
#             examples = examples_new
#             batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
#     #         else:
#     #             batch = {"input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

#             # If special token mask has been preprocessed, pop it from the dict.
#             batch["next_sentence_label"] = torch.tensor([_["next_sentence_label"] for _ in examples])
#             return batch
#         else:
#             batch = {"special_tokens_mask":[],
#                   "attention_mask":[],
#                   "token_type_ids":[],
#                   "input_ids": []}
#             batch_size = len(examples)
#             next_sentence_label = torch.randint(0,2,[batch_size]).detach()
#             if batch_size > 1:
#                 pre_i_add = torch.randint(1,batch_size, [batch_size]).detach()
#             else:
#                 pre_i_add = torch.randint(0,batch_size, [batch_size]).detach()
#             example_sentences = []
#             previous_sentences = []
#             for i, example in enumerate(examples):
#                 if next_sentence_label[i] == 0:
#                     # here we need to feed a positive sample
#                     if example["previous_sentence_label"] == 1:
#                         previous_sentence = example["previous_sentence"]
#                     else:
#                         # if we could not find the positive sample, we change the label
#                         next_sentence_label[i] = 1
#                 if next_sentence_label[i] == 1:
#                     # need to pick a negative sample
#                     pre_i = (i + int(pre_i_add[i]))%batch_size
#                     if batch_size > 1:
#                         assert pre_i != i
#                         while ((examples[pre_i]["id"] == example["id"] 
#                                and examples[pre_i]["joined_all_par_trans_index"] == example["joined_all_par_trans_index"] -1 )
#                                or pre_i == i):
#                             pre_i = (pre_i + 1)%batch_size
#                         previous_sentence = examples[pre_i]["joined_all_par_trans"]
#                     else:
#                         previous_sentence = example["joined_all_par_trans"]
#                 example_sentences.append(example["joined_all_par_trans"])
#                 previous_sentences.append(previous_sentence)
#             tokenized_samples = self.tokenizer(previous_sentences,example_sentences,padding=False,truncation=True,
#                 max_length=self.tokenizer.model_max_length,
#                 # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
#                 # receives the `special_tokens_mask`.
#                 return_special_tokens_mask=True,
#             )
#             batch = self.tokenizer.pad(tokenized_samples, return_tensors="pt")
#             batch["next_sentence_label"] = next_sentence_label
#         special_tokens_mask = batch.pop("special_tokens_mask", None)
#         if self.mlm:
#                 batch["input_ids"], batch["labels"] = self.mask_tokens(
#                     batch["input_ids"], special_tokens_mask=special_tokens_mask
#                 )
#         else:
#             labels = batch["input_ids"].clone()
#             if self.tokenizer.pad_token_id is not None:
#                 labels[labels == self.tokenizer.pad_token_id] = -100
#             batch["labels"] = labels
#         return batch
    
class MyDataset(Dataset):
    def __init__(self, data_file, header='infer', delimiter=","):
        assert data_file.split(".")[-1] == "csv"
        self.df = pd.read_csv(data_file,header = header, delimiter=delimiter)
        
    def __len__(self):
        return len(self.df)
   
    def __getitem__(self, idx):
        assert idx < self.__len__()
        return self.df.iloc[idx]

@dataclass
class MyDataCollator:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    data_type: str = "train"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
            
        self.generator = torch.Generator()
        self.generator.manual_seed(self.str2int(self.__class__.__name__ + self.data_type))
    
    def str2int(self,s):
        ord_s = [str(ord(_)) for _ in s]
        return int("".join(ord_s))%10000007
    
    def reset_generator_manual_seed(self, epoch=0):
        self.generator.manual_seed(self.str2int(self.__class__.__name__ + self.data_type) + epoch)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]): # -> Dict[str, torch.Tensor]
        # Handle dict or lists with proper padding and conversion to tensor.
        if "ad" not in examples[0].index:
            logger.info("To use the pre-defined negative samples")
            # we use the pre-defined data for training, so the NSP training data will be the same for all epochs
#             examples_new = [{k:v for k,v in _.items() if k not in ["id", "joined_all_par_trans", "joined_all_par_trans_index", 
#                                                   "previous_sentence", "previous_sentence_label"]} 
#                        for _ in examples]
#            examples = examples_new
            previous_sentences = [_.previous_sentence for _ in examples]
            example_sentences = [_.joined_all_par_trans for _ in examples]
            tokenized_samples = self.tokenizer(previous_sentences,example_sentences,padding=False,truncation=True,
                max_length=self.tokenizer.model_max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
            
            batch = self.tokenizer.pad(tokenized_examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
            batch["next_sentence_label"] = torch.tensor([1 - _["previous_sentence_label"] for _ in examples])
            return batch
        else:
            batch = {"special_tokens_mask":[],
                  "attention_mask":[],
                  "token_type_ids":[],
                  "input_ids": []}
            batch_size = len(examples)
            next_sentence_label = torch.randint(0,2,[batch_size], generator=self.generator).detach()
            if batch_size > 1:
                pre_i_add = torch.randint(1,batch_size, [batch_size], generator=self.generator).detach()
            else:
                pre_i_add = torch.randint(0,batch_size, [batch_size], generator=self.generator).detach()
            example_sentences = []
            previous_sentences = []
            for i, example in enumerate(examples):
                if next_sentence_label[i] == 0:
                    # here we need to feed a positive sample
                    if example["previous_sentence_label"] == 1:
                        previous_sentence = example["previous_sentence"]
                    else:
                        # if we could not find the positive sample, we change the label
                        next_sentence_label[i] = 1
                if next_sentence_label[i] == 1:
                    # need to pick a negative sample
                    pre_i = (i + int(pre_i_add[i]))%batch_size
                    if batch_size > 1:
                        assert pre_i != i
                        # print('examples[pre_i]["joined_all_par_trans_index"]', examples[pre_i]["joined_all_par_trans_index"])
                        # print('example["joined_all_par_trans_index"] -1', example["joined_all_par_trans_index"] -1)
                        if ((examples[pre_i]["id"] == example["id"] 
                               and examples[pre_i]["joined_all_par_trans_index"] == example["joined_all_par_trans_index"] -1 )):
                            pre_i = (pre_i + 1)%batch_size
                        previous_sentence = examples[pre_i]["joined_all_par_trans"]
                    else:
                        previous_sentence = example["joined_all_par_trans"]
                example_sentences.append(example["joined_all_par_trans"])
                previous_sentences.append(previous_sentence)
            tokenized_samples = self.tokenizer(previous_sentences,example_sentences,padding=False,truncation=True,
                max_length=self.tokenizer.model_max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
            batch = self.tokenizer.pad(tokenized_samples, return_tensors="pt")
            batch["next_sentence_label"] = next_sentence_label
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
                batch["input_ids"], batch["labels"] = self.mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ): #  -> Tuple[torch.Tensor, torch.Tensor]
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix,generator=self.generator).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8), generator=self.generator).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5),generator=self.generator).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class MyMlmDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length. Changed to load only MLM tuning data

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    data_type: str = "train"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
            
        self.generator = torch.Generator()
        self.generator.manual_seed(self.str2int(self.__class__.__name__ + self.data_type))
    
    def str2int(self,s):
        ord_s = [str(ord(_)) for _ in s]
        return int("".join(ord_s))%10000007
    
    def reset_generator_manual_seed(self, epoch=0):
        self.generator.manual_seed(self.str2int(self.__class__.__name__ + self.data_type) + epoch)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]): #  -> Dict[str, torch.Tensor]
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {
            "input_ids": [], # _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of),
        }

        example_sentences = []
        for i, example in enumerate(examples):
            example_sentences.append(example["joined_all_par_trans"])
        tokenized_samples = self.tokenizer(example_sentences,padding=False,truncation=True,
                max_length=self.tokenizer.model_max_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        batch = self.tokenizer.pad(tokenized_samples, return_tensors="pt")
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ): #  -> Tuple[torch.Tensor, torch.Tensor]
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix,generator=self.generator).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8), generator=self.generator).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5),generator=self.generator).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels