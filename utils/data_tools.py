# -*- coding:utf-8 -*-

import json

def read_data_json(filename):
    with open(filename, 'rt', encoding='utf-8') as file:
        return json.load(file)

def pad_sen(sen_idx_list, max_len=115, pad_idx=1):
    return sen_idx_list + [pad_idx] * (max_len - len(sen_idx_list))

def string_2_idx_sen(sen,  vocab_dict):
    if "UNK_token" in vocab_dict.keys():
        unk_idx = vocab_dict['UNK_token']
    idx_sen = [vocab_dict[word] if word in vocab_dict.keys() else unk_idx for word in sen]
    return idx_sen
