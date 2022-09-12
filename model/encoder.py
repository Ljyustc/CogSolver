# -*- coding:utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from .attention import get_mask, Attention

class PositionalEncoding(nn.Module):
    def __init__(self, pos_size, dim):
        super(PositionalEncoding, self).__init__()
        pe = torch.rand(pos_size, dim)
        # (0, 1) => (-1, 1)
        pe = pe * 2 - 1
        self.pe = nn.Parameter(pe)
    
    def forward(self, input):
        output = input + self.pe[:input.size(1)]
        return output

class Encoder(nn.Module):
    def __init__(self, embed_model, hidden_size=512, word_know_threshold=0.5, span_size=0, dropout=0.4):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.word_know_threshold = word_know_threshold
        embed_size = embed_model.embedding_dim
        
        self.embedding = embed_model
        # word encoding
        self.word_rnn = nn.GRU(embed_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        # span encoding
        # span sequence
        self.span_attn = Attention(self.hidden_size, mix=True, fn=True) 
        self.pos_enc = PositionalEncoding(span_size, hidden_size)
        # merge subtree/word node
        self.to_parent = Attention(self.hidden_size, mix=True, fn=True)

    def bi_combine(self, output, hidden):
        # combine forward and backward LSTM
        # (num_layers * num_directions, batch, hidden_size).view(num_layers, num_directions, batch, hidden_size)
        hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden
    
    def dependency_encode(self, word_output, tree):
        word, rel, left, right = tree
        children = left + right
        word_vector = word_output[:, word]
        if len(children) == 0:
            vector = word_vector
        else:
            children_vector = [self.dependency_encode(word_output, child).unsqueeze(1) for child in children]
            children_vector = torch.cat(children_vector, dim=1)
            query = word_vector.unsqueeze(1)
            vector = self.to_parent(query, children_vector)[0].squeeze(1)
        return vector
    
    def convert_tree_to_adj(self, tree, mat):
        root, _, left, right = tree
        left_child = [x[0] for x in left]
        right_child = [x[0] for x in right]
        child = left_child + right_child
        if len(child) == 0:
            return
        else:
            mat[root,child] = 1
            if len(left_child) != 0:
                for item in left:
                    self.convert_tree_to_adj(item, mat)
            if len(right_child) != 0:
                for item in right:
                    self.convert_tree_to_adj(item, mat)
                
    def forward(self, input_var, input_lengths, span_length, knowledge_base, tree=None):
        use_cuda = span_length.is_cuda
        pad_hidden = torch.zeros(1, self.hidden_size)
        if use_cuda:
            pad_hidden = pad_hidden.cuda()
        
        word_outputs = []
        span_inputs = []
        word_operators = []
        depend_relation = []
        word_init = []
        
        input_vars = input_var
        trees = tree  # list(max_span_len) of batch_size trees
        bi_word_hidden = None
        for span_index, input_var in enumerate(input_vars):
            input_length = input_lengths[span_index]

            # word encoding
            init_embedded = self.embedding(input_var)
            word_init.append(init_embedded)
            
            init_embedded_drop = self.dropout(init_embedded)
            # at least 1 word in some full padding span
            pad_input_length = input_length.clone()
            pad_input_length[pad_input_length == 0] = 1
            embedded = nn.utils.rnn.pack_padded_sequence(init_embedded_drop, pad_input_length.cpu(), batch_first=True, enforce_sorted=False)
            word_output, bi_word_hidden = self.word_rnn(embedded, bi_word_hidden)
            word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)
            word_output, word_hidden = self.bi_combine(word_output, bi_word_hidden)
            
            # word-operator
            word_operator = knowledge_base(init_embedded)  # size: (batch_size, words, op)
            word_operators.append(word_operator)
            
            word_maxlen = input_var.size(1)
        
            # tree encoding
            span_span_input = []
            word_word_adj = []
            for data_index, data_word_output in enumerate(word_output):
                data_word_output = data_word_output.unsqueeze(0)
                tree = trees[span_index][data_index]
                # calculate dependency
                depend_adj = np.zeros((word_maxlen, word_maxlen), dtype=np.float32)  
                if tree is not None:
                    data_span_input = self.dependency_encode(data_word_output, tree)
                    self.convert_tree_to_adj(tree, depend_adj)
                else:
                    data_span_input = pad_hidden
                depend_adj = torch.from_numpy(depend_adj + depend_adj.T + np.eye(word_maxlen, dtype=np.float32))  # undirected, self-loop
                depend_adj = Variable(depend_adj.cuda()) if use_cuda else Variable(depend_adj)
                span_span_input.append(data_span_input)
                word_word_adj.append(depend_adj.unsqueeze(0))
            span_input = torch.cat(span_span_input, dim=0)
            span_inputs.append(span_input.unsqueeze(1))
            word_outputs.append(word_output)
            depend_relation.append(torch.cat(word_word_adj, dim=0))  
        # depend_relation: list(max_span), each shape: (batch_size, words, words)
        
        # word-operator relation 
        word_operator = torch.cat(word_operators, dim=1)  # size: (batch_size, all_words, opt)
        # word-word relation 
        all_word = torch.cat(word_init, dim=1)
        dis = -torch.norm(all_word.unsqueeze(-2)-all_word.unsqueeze(1),p=2,dim=-1) + torch.mean(torch.norm(all_word.unsqueeze(-2)-all_word.unsqueeze(1),p=2,dim=-1))
        word_word = torch.sigmoid(dis)  # size: (batch_size, all_words, all_words)
        word_word = word_word.masked_fill(word_word < self.word_know_threshold, 0)
        # span encoding
        span_input = torch.cat(span_inputs, dim=1)
        span_mask = get_mask(span_length, span_input.size(1))
        span_output = self.pos_enc(span_input)
        span_output = self.dropout(span_output)
        span_output, _ = self.span_attn(span_output, span_output, span_mask)
        span_output = span_output * (span_mask == 0).unsqueeze(-1)
        dim0 = torch.arange(span_output.size(0))
        if use_cuda:
            dim0 = dim0.cuda()
        span_hidden = span_output[dim0, span_length - 1].unsqueeze(0)
        return [span_output, word_outputs], [word_operator, word_word, depend_relation], span_hidden
