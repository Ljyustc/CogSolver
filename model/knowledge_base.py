# -*- coding:utf-8 -*-

import torch
from torch import nn
import numpy as np
import math

class Knowledge_Base(nn.Module):
    def __init__(self, embed_model, op_set, vocab_dict, op_threshold=0.5, use_cuda=True):
        super(Knowledge_Base, self).__init__()
        self.use_cuda = use_cuda
        self.embed_size = embed_model.embedding_dim
        self.op_set = op_set
        self.op_size = len(op_set)
        self.get_embedding_meta(vocab_dict)
        self.embed_model = embed_model
        self.op_threshold = op_threshold
        
    def get_embedding_meta(self, vocab_dict):
        self.op_vocab = torch.LongTensor([vocab_dict[token] for token in list(self.op_set)])
        if self.use_cuda:
            self.op_vocab = self.op_vocab.cuda()
        return
    
    def forward(self, word_output):
        op_init_embedding = self.embed_model(self.op_vocab)
        dis = -torch.norm(word_output.unsqueeze(-2)-op_init_embedding, p=2, dim=-1) 
        prob = torch.softmax(dis, dim=-1)
        prob = prob.masked_fill(prob < self.op_threshold, 0)
        return prob
