import os
from random import sample
import sys
import numpy as np
import torch
import pandas as pd
# from transformers import AutoTokenizer
import transformers as ppb
import re

def load_transformer_model_tokenizer(model_class, tokenizer_class, pretrained_weights):
    # ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return tokenizer, model

def simple_split_sentences(text):
    '''split sentences'''
    # text_write = re.split(r"(\(\.+\)|\+\.+|\.|\?)", text)

    text_write = re.split(r"(\.|\?|\。|\,|\，)", text) 

    values = text_write[::2]
    delimiters = text_write[1::2]

    output = [values[0]]
    
    i = 0
    while (i < len(delimiters)):
        if  len(delimiters[i]) > 1:
            output[-1] += delimiters[i] + values[i+1]
        else:    
            output[-1] += delimiters[i]
            output.append(values[i+1])
        i += 1

    return output

def window_tokenizer(subject_text, tokenizer, max_len):
    max_token_len = max_len-2
    tokenized_windows = []
    current_window = []
    for sent_text in subject_text:
        sent_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_text))
        if len(current_window) + len(sent_tokenized) <= max_token_len:
            current_window += sent_tokenized
        else:
            tokenized_windows.append(tokenizer.encode(current_window, add_special_tokens=True, max_length=max_len, truncation=True))
            current_window = []
    if len(current_window) > 0:
        tokenized_windows.append(tokenizer.encode(current_window, add_special_tokens=True, max_length=max_len, truncation=True))
    return tokenized_windows

def window_token_based_cut(subject_text, tokenizer, max_len):
    '''cut trans tokens into windows'''
    max_token_len = max_len-2
    tokenized_windows = []
    current_window = []
    total_text = []
    window_text = ''
    for sent_text in subject_text:
        sent_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_text))
        if len(current_window) + len(sent_tokenized) <= max_token_len:
            current_window += sent_tokenized
            window_text += sent_text
        else:
            tokenized_windows.append(tokenizer.encode(current_window, add_special_tokens=True, max_length=max_len, truncation=True))
            current_window = []
            total_text.append(window_text)
            window_text = ''

    if len(current_window) > 0:
        tokenized_windows.append(tokenizer.encode(current_window, add_special_tokens=True, max_length=max_len, truncation=True))
        total_text.append(window_text)
    return total_text

def read_input_text_len_control(df, sample_size=-1, max_len=128, token_cut=True):    
    if token_cut:
        # tokenizer = AutoTokenizer.from_pretrained('/project_bdda5/bdda/ywang/class_ncd/new_models/bert-base-uncased')
        tokenizer, _ = load_transformer_model_tokenizer(ppb.BertModel, ppb.BertTokenizer, '/project_bdda5/bdda/ywang/class_ncd/new_models/bert-base-uncased')
    else:
        raise NotImplemented
    
    def pre_tokenizing(text):
        if isinstance(text, str):
            text = [y + '.' for y in text.strip('.').split('.')]
        else:
            raise ValueError('wrong type of text read from chas csv')

        return window_tokenizer(text, tokenizer, max_len)

    pre_tokened_series = df.joined_all_par_trans.apply(lambda x: pre_tokenizing(x))

    texts_df = df[pre_tokened_series.apply(lambda x: True if len(x) == 1 else False)]

    if sample_size==-1:
        output_df = texts_df[["id", "joined_all_par_trans", "ad"]]
        return output_df.reset_index(drop=True)
    else:
        all_number = len(texts_df)
        indexes_list = [[i*sample_size, (i+1)*sample_size] for i in range(all_number // sample_size)] # + [[all_number-all_number%sample_size, all_number]]
        list_output_dfs = [texts_df[["id", "joined_all_par_trans", "ad"]].iloc[i1:i2, :].reset_index(drop=True) for [i1, i2] in indexes_list]
        return list_output_dfs

def read_input_no_len_control(df, model, sample_size=-1, max_len=512, save_trans=None):    
    if save_trans is not None:
        if 'bert' in model and 'chinese' not in model:
            tokenizer, _ = load_transformer_model_tokenizer(ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        elif 't5' in model:
            tokenizer, _ = load_transformer_model_tokenizer(ppb.T5Model, ppb.T5Tokenizer, 't5-base')
        # elif 'gpt' in model:
        #     tokenizer, _ = load_transformer_model_tokenizer(ppb.GPT2Model, ppb.GPT2Tokenizer, 'gpt2')
        else:
            raise ValueError

        def pre_tokenizing(text):
            '''split into sentences -> tokenize'''
            if isinstance(text, str):
                text = simple_split_sentences(text.strip('.'))
            else:
                raise ValueError('wrong type of text read from chas csv')

            return window_token_based_cut(text, tokenizer, max_len)[0]

        df.joined_all_par_trans = df.joined_all_par_trans.apply(lambda x: pre_tokenizing(x))    

    # texts_df = df[pre_tokened_series.apply(lambda x: True if len(x) == 1 else False)]

    if sample_size==-1:
        output_df = df[["id", "joined_all_par_trans", "ad", "age"]] # "age_group", 
        if save_trans is not None:
            output_df.to_csv(save_trans)
        return output_df.reset_index(drop=True)
    else:
        raise NotImplemented
        # all_number = len(df)
        # indexes_list = [[i*sample_size, (i+1)*sample_size] for i in range(all_number // sample_size)] # + [[all_number-all_number%sample_size, all_number]]
        # list_output_dfs = [df[["id", "joined_all_par_trans", "ad", "age"]].iloc[i1:i2, :].reset_index(drop=True) for [i1, i2] in indexes_list] # "age_group",
        # return list_output_dfs