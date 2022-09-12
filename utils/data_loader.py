# -*- coding:utf-8 -*-

import json
import logging
import os
import torch
from .data_tools import read_data_json, string_2_idx_sen, pad_sen
from .nlp_tools import parse_dependency_tree

def convert_dependency(dataset):
    for item in dataset:
        item["dependency"] = [parse_dependency_tree(json.loads(str_dependency)) for str_dependency in item["dependency"]]
    return

class DataLoader():
    def __init__(self, args, trim_min_count=0, embed_dim=128, build_wv=False):
        self.args = args

        train_list = read_data_json("./data/train23k_processed.json")
        valid_list = read_data_json("./data/valid23k_processed.json")
        test_list = read_data_json("./data/test23k_processed.json")
        self.train_list = train_list
        self.test_list = test_list
        self.wv_path = f"./data/word2vec.pt"
        self.op_set = set("+-*/^")
        logging.info(f"train size: {len(self.train_list)}, test size: {len(self.test_list)}")     

        convert_dependency(self.train_list)
        convert_dependency(self.test_list)

        build_wv = not os.path.exists(self.wv_path)
        embed_vectors, self.vocab_list, self.class_list, self.span_size = self.preprocess_vocab(embed_dim, trim_min_count, build_wv=build_wv)
        if build_wv:
            self.embed_vectors = torch.tensor(embed_vectors)
            torch.save(self.embed_vectors, self.wv_path)
        else:
            self.embed_vectors = torch.load(self.wv_path)
        
        self.vocab_dict = {token: idx for idx, token in enumerate(self.vocab_list)}
        self.vocab_len = len(self.vocab_list) 
        self.class_dict = {token: idx for idx, token in enumerate(self.class_list)}

        self.test_list = self.preprocess_dataset(self.test_list)
        if self.args.run_flag == "train":
            self.train_list = self.preprocess_dataset(self.train_list)
        return

    def preprocess_vocab(self, embed_dim, trim_min_count, build_wv=False):
        # word count
        sentences = []
        word_count ={}
        equ_tokens = set()
        num_tokens = set()

        span_size = max(len(data["spans"]) for data in self.train_list + self.test_list)

        for data in self.train_list:
            sentence = [word for span in data["spans"] for word in span.strip().split(' ')]
            sentence = [word if "temp_" not in word else 'NUM_token' for word in sentence]
            sentences.append(sentence)
            for word in sentence:
                word_count[word] = word_count.get(word, 0) + 1
        
        for data in self.train_list + self.test_list:
            equation = data['target_norm_pre_template']
            if type(equation) is str:
                equation = equation.split(' ')
            equation = equation[2:]
            for token in equation:
                if token not in equ_tokens:
                    equ_tokens.add(token)

        num_tokens = set(token for data in self.train_list + self.test_list for span in data["spans"] for token in span.strip().split(' ') if "temp_" in token)

        if build_wv:
            from gensim.models import word2vec
            import numpy as np
            model = word2vec.Word2Vec(sentences, vector_size=embed_dim, min_count=1)

        vocab_list = ['PAD_token', 'UNK_token', 'END_token']
        class_list = ['PAD_token', 'END_token']
        if build_wv:
            embed_vectors = []
            embed_vectors.append(np.zeros(embed_dim))           # PAD_token
            embed_vectors.append(np.random.rand(embed_dim))     # UNK_token
            embed_vectors.append(np.random.rand(embed_dim))     # END_token

        for word, cnt in sorted(word_count.items(), key=lambda x: x[1]):
            if cnt >= trim_min_count:
                vocab_list.append(word)
                if build_wv:
                    embed_vectors.append(np.array(model.wv[word]))

        for token in sorted(equ_tokens):
            class_list.append(token)
        
        for token in sorted(num_tokens):
            if token not in class_list:
                class_list.append(token)
        class_list.sort()
        
        logging.info(f"saving vocab({trim_min_count}): {len(vocab_list) - 3} / {len(word_count)} = {(len(vocab_list) - 3)/len(word_count)}")
        
        for token in class_list:
            if token not in vocab_list:
                vocab_list.append(token)
                if build_wv:
                    embed_vectors.append(np.random.rand(embed_dim))
        
        if build_wv:
            embed_vectors = np.array(embed_vectors)
        else:
            embed_vectors = None

        logging.info(f"vocab_len: {len(vocab_list)}, classes_len: {len(class_list)}")
        logging.info(f"decode_classes: {' '.join(class_list)}")
        logging.info(f"max_span_len: {span_size}")
        return embed_vectors, vocab_list, class_list, span_size

    def preprocess_dataset(self, dataset):
        data_dataset = []
        for data in dataset:
            # index
            index = data['id']
            # num_list
            num_list = data['num_list']
            # answer
            solution = data['answer']
            # text
            encode_text = ' '.join(data["spans"])

            # span
            raw_spans = [span.strip().split(' ') for span in data["spans"]]
            encode_spans = [[word if "temp_" not in word else 'NUM_token' for word in span] for span in raw_spans]
            encode_spans_idx = [string_2_idx_sen(span, self.vocab_dict) for span in encode_spans]
            encode_spans_len = [len(span) for span in encode_spans]
            span_len = len(encode_spans)
            
            # target
            decode_sen = data['target_norm_pre_template']
            if type(decode_sen) is str:
                decode_sen = decode_sen.split(' ')
            decode_sen = decode_sen[2:]
            decode_sen.append('END_token')
            decode_sen_idx = string_2_idx_sen(decode_sen, self.class_dict)
            decode_len = len(decode_sen_idx)
            
            # num_pos, span_num_pos, word_num_pos
            span_num_pos = [-1] * len(self.class_list)
            word_num_poses = [[-1] * len(self.class_list) for _ in range(len(raw_spans))]
            for i, span in enumerate(raw_spans):
                for j, word in enumerate(span):
                    if "temp_" in word and word in self.class_dict.keys():
                        class_index = self.class_dict[word]
                        span_num_pos[class_index] = i
                        word_num_poses[i][class_index] = j
        
            data_elem = dict()
            data_elem['index'] = index
            data_elem['text'] = encode_text
            data_elem['num_list'] = num_list
            data_elem['solution'] = solution

            data_elem['span_encode_idx'] = encode_spans_idx
            data_elem['span_encode_len'] = encode_spans_len
            data_elem['span_len'] = span_len
            
            data_elem['span_num_pos'] = span_num_pos
            data_elem['word_num_poses'] = word_num_poses
            
            data_elem['tree'] = data["dependency"]
            
            data_elem['decode_idx'] = decode_sen_idx
            data_elem['decode_len'] = decode_len
            
            data_dataset.append(data_elem)
        return data_dataset
          
    def _data_batch_preprocess(self, data_batch, template_flag):
        batch_idxs = []
        batch_text = []
        batch_num_list = []
        batch_solution = []

        batch_span_encode_idx = []
        batch_span_encode_len = []
        batch_span_len = []

        batch_decode_idx = []
        batch_decode_len = []

        batch_span_num_pos = []
        batch_word_num_poses = []
        batch_tree = []

        for data in data_batch:
            # id
            batch_idxs.append(data["index"])
            # text
            batch_text.append(data["text"])
            # num_list
            batch_num_list.append(data['num_list'])
            # answer
            batch_solution.append(data['solution'])

            # spans
            # sample * span
            batch_span_encode_idx.append(data["span_encode_idx"])
            batch_span_encode_len.append(data["span_encode_len"])
            batch_span_len.append(data["span_len"])
            
            if template_flag:
                # target
                batch_decode_idx.append(data["decode_idx"])
                batch_decode_len.append(data["decode_len"])

            # num_pos, span_num_pos, word_num_poses
            batch_span_num_pos.append(data["span_num_pos"])
            batch_word_num_poses.append(data["word_num_poses"])
            
            # dependency
            batch_tree.append(data["tree"])

        # pad
        # max_len
        max_span_len = max(batch_span_len)
        max_span_encode_len = []
        for i in range(max_span_len):
            max_encode_len = max(elem_span_len[i] for elem_span_len in batch_span_encode_len if i < len(elem_span_len))
            max_span_encode_len.append(max_encode_len)
        if template_flag:
            max_decode_len = max(batch_decode_len)
        
        # span * sample
        batch_span_encode_idx_pad = [[] for _ in range(max_span_len)]
        batch_span_encode_len_pad = [[] for _ in range(max_span_len)]       
        batch_word_num_poses_pad = [[] for _ in range(max_span_len)]
        batch_tree_pad = [[] for _ in range(max_span_len)]
        # decode
        if template_flag:
            batch_decode_idx_pad = []

        pad_spans = [[self.vocab_dict['PAD_token']] * encode_len for encode_len in max_span_encode_len]
        pad_num_pos = [-1] * len(self.class_list)
        for data_index in range(len(data_batch)):
            # spans
            span_len = batch_span_len[data_index]
            encode_spans = batch_span_encode_idx[data_index]
            encode_lens = batch_span_encode_len[data_index]
            for span_index in range(max_span_len):
                max_encode_len = max_span_encode_len[span_index]
                if span_index < span_len:
                    encode_span = encode_spans[span_index]
                    encode_span = pad_sen(encode_span, max_encode_len, self.vocab_dict['PAD_token'])
                    encode_len = encode_lens[span_index]
                else:
                    encode_span = pad_spans[span_index]
                    encode_len = 0
                batch_span_encode_idx_pad[span_index].append(encode_span)
                batch_span_encode_len_pad[span_index].append(encode_len)
            
            if template_flag:
                # target
                decode_sen_idx = batch_decode_idx[data_index]
                decode_sen_idx_pad = pad_sen(decode_sen_idx, max_decode_len, self.class_dict['PAD_token'])
                batch_decode_idx_pad.append(decode_sen_idx_pad)
            
            # word_num_poses
            word_num_poses = batch_word_num_poses[data_index]
            for span_index in range(max_span_len):
                if span_index < span_len:
                    word_num_pos = word_num_poses[span_index]
                else:
                    word_num_pos = pad_num_pos
                batch_word_num_poses_pad[span_index].append(word_num_pos)
            
            # dependency
            trees = batch_tree[data_index]
            for span_index in range(max_span_len):
                if span_index < span_len:
                    tree = trees[span_index]
                else:
                    tree = None
                batch_tree_pad[span_index].append(tree)

        batch_data_dict = dict()
        batch_data_dict['batch_index'] = batch_idxs
        batch_data_dict['batch_text'] = batch_text
        batch_data_dict['batch_num_list'] = batch_num_list
        batch_data_dict['batch_solution'] = batch_solution

        batch_data_dict["batch_span_encode_idx"] = batch_span_encode_idx_pad
        batch_data_dict["batch_span_encode_len"] = batch_span_encode_len_pad
        batch_data_dict["batch_span_len"] = batch_span_len

        batch_data_dict["batch_span_num_pos"] = batch_span_num_pos
        batch_data_dict["batch_word_num_poses"] = batch_word_num_poses_pad
        
        batch_data_dict["batch_tree"] = batch_tree_pad

        if template_flag:
            batch_data_dict['batch_decode_idx'] = batch_decode_idx_pad
        return batch_data_dict

    def get_batch(self, data_list, batch_size, template_flag = False):
        batch_num = int(len(data_list)/batch_size)
        if len(data_list) % batch_size != 0:
            batch_num += 1
        for idx in range(batch_num):
            batch_start = idx*batch_size
            batch_end = min((idx+1)*batch_size, len(data_list))
            batch_data_dict = self._data_batch_preprocess(data_list[batch_start: batch_end], template_flag)
            yield batch_data_dict
