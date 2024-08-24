# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from torch.nn.parameter import Parameter
from transformers import BertModel, BertTokenizer


class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size, dropout=0.5):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)

    def forward(self, input_seqs):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        return embedded


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:,
                                                             :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs),
                              2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(
            self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(
            max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size,
                          hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(
            1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(
            last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(
            torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(
            torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings),
                              2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs),
                              2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(
            max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class BertEncoder(nn.Module):
    def __init__(self, bert_model, dropout=0.5):
        super(BertEncoder, self).__init__()
        self.bert_layer = BertModel.from_pretrained(bert_model)
        self.em_dropout = nn.Dropout(dropout)

    def forward(self, input_ids, knowledge_base, attention_mask=None):
        output = self.bert_layer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embedded = self.em_dropout(output[0]) # B x S x Bert_emb(768)
        
        dis = -torch.norm(embedded.unsqueeze(-2)-embedded.unsqueeze(1),p=2,dim=-1) + torch.mean(torch.norm(embedded.unsqueeze(-2)-embedded.unsqueeze(1),p=2,dim=-1))
        word_word = torch.sigmoid(dis)  # B x S x S
        word_word = word_word.masked_fill(word_word < knowledge_base.word_threshold, 0)
        word_operator = knowledge_base(embedded)  # B x S x O
        return embedded, word_word, word_operator


class EncoderSeq(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru_pade = nn.LSTM(embedding_size, hidden_size,
                               n_layers, dropout=dropout, bidirectional=True)

    def forward(self, embedded, input_lengths, knowledge_base, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        embedded1 = embedded.transpose(0,1)  # B x S x E
        dis = -torch.norm(embedded1.unsqueeze(-2)-embedded1.unsqueeze(1),p=2,dim=-1) + torch.mean(torch.norm(embedded1.unsqueeze(-2)-embedded1.unsqueeze(1),p=2,dim=-1))
        word_word = torch.sigmoid(dis)  # B x S x S
        word_word = word_word.masked_fill(word_word < knowledge_base.word_threshold, 0)
        word_operator = knowledge_base(embedded1)  # B x S x O
        
        problem_output = pade_outputs[-1, :, :self.hidden_size] + \
            pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + \
            pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output, word_word, word_operator

class EncoderBert(nn.Module):
    def __init__(self, hidden_size, bert_pretrain_path='', dropout=0.5):
        super(EncoderBert, self).__init__()
        self.embedding_size = 768
        print("bert_model: ", bert_pretrain_path)
        self.bert_model = BertModel.from_pretrained(bert_pretrain_path)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.em_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)

    def forward(self, input_ids, knowledge_base, attention_mask=None):
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embedded = self.em_dropout(output[0]) # B x S x Bert_emb(768)
        
        dis = -torch.norm(embedded.unsqueeze(-2)-embedded.unsqueeze(1),p=2,dim=-1) + torch.mean(torch.norm(embedded.unsqueeze(-2)-embedded.unsqueeze(1),p=2,dim=-1))
        word_word = torch.sigmoid(dis)  # B x S x S
        word_word = word_word.masked_fill(word_word < knowledge_base.word_threshold, 0)
        word_operator = knowledge_base(embedded)  # B x S x O
        
        pade_outputs = self.linear(embedded) # B x S x E
        pade_outputs = pade_outputs.transpose(0,1) # S x B x E

        problem_output = pade_outputs[0]
        return pade_outputs, problem_output, word_word, word_operator

class Knowledge_Base(nn.Module):
    def __init__(self, embed_model, op_emb, vocab_dict, word_threshold=0.7, op_threshold=0.3, USE_CUDA=True, fold=0):
        super(Knowledge_Base, self).__init__()
        self.embedding_size = 768
        self.op_emb = op_emb
        self.op_size, self.hidden_size = op_emb.size()
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)
        
        self.use_cuda = USE_CUDA
        self.get_embedding_meta(vocab_dict)
        self.embed_model = embed_model
        self.word_threshold = word_threshold
        self.op_threshold = op_threshold
        
        self.ww_know_save = "fold_"+str(fold)+'word_word.pt'
        self.wo_know_save = "fold_"+str(fold)+'word_op.pt'
        self.get_know_init(vocab_dict, self.op_size)

    def get_embedding_meta(self, vocab_dict):
        self.word_vocab = torch.LongTensor(range(len(vocab_dict)))
        if self.use_cuda:
            self.word_vocab = self.word_vocab.cuda()
        return

    def get_know_init(self, vocab_dict, op_size):
        if not os.path.exists(self.ww_know_save) or not os.path.exists(self.wo_know_save):
            ww = torch.rand(len(vocab_dict), len(vocab_dict))
            wo = torch.softmax(torch.randn(len(vocab_dict), op_size), dim=-1)
            torch.save(ww, self.ww_know_save)
            torch.save(wo, self.wo_know_save)
            
        self.delta_word_word = torch.zeros((len(vocab_dict), len(vocab_dict)))
        self.delta_word_op = torch.zeros((len(vocab_dict), op_size))

    def get_delta_know(self, word_id):
        # word_id: batch x word_len
        ww = self.delta_word_word
        wo = self.delta_word_op
        ww_ = torch.cat([ww[x][:,x].unsqueeze(0) for x in word_id],dim=0)  # batch x word_len x word_len
        wo_ = torch.cat([wo[x].unsqueeze(0) for x in word_id],dim=0)
        if self.use_cuda:
            ww_ = ww_.cuda()
            wo_ = wo_.cuda()
        return ww_, wo_
        
    def get_know(self, word_id):
        # word_id: batch x word_len
        # ww: word_num x word_num
        ww = torch.load(self.ww_know_save)
        wo = torch.load(self.wo_know_save)
        ww_ = torch.cat([ww[x][:,x].unsqueeze(0) for x in word_id],dim=0)  # batch x word_len x word_len
        wo_ = torch.cat([wo[x].unsqueeze(0) for x in word_id],dim=0)
        if self.use_cuda:
            ww_ = ww_.cuda()
            wo_ = wo_.cuda()
        return ww_, wo_
    
    def get_know_from(self, word_id, ww, wo):
        # word_id: batch x word_len
        ww_ = torch.cat([ww[x][:,x].unsqueeze(0) for x in word_id],dim=0)  # batch x word_len x word_len
        wo_ = torch.cat([wo[x].unsqueeze(0) for x in word_id],dim=0)
        if self.use_cuda:
            ww_ = ww_.cuda()
            wo_ = wo_.cuda()
        return ww_, wo_
    
    def cal_know(self):
        batch = 32  # calculate knowledge in batch for GPU storage
        word_num = len(self.word_vocab)
        num_batch = int(np.ceil(word_num/batch))
        
        word_word_all = []
        all_word = self.embed_model(self.word_vocab)  # word_num x d
        for i in range(num_batch):
            batch_input = self.word_vocab[i*batch: min((i+1)*batch, word_num)]
            batch_embed = self.embed_model(batch_input)  # batch x d
            dis = torch.norm(batch_embed.unsqueeze(1)-all_word.unsqueeze(0),p=2,dim=-1)  # batch x word_num
            
            word_word_all.append(dis)
        dis = torch.cat(word_word_all, dim=0)
        mean_dis = torch.mean(dis)
        dis = -dis + mean_dis
        word_word = torch.sigmoid(dis)  # word_num x word_num
        word_word = word_word.masked_fill(word_word < self.word_threshold, 0)

        dis = -torch.norm(self.linear(all_word).unsqueeze(1)-self.op_emb, p=2, dim=-1) 
        word_op = torch.softmax(dis, dim=-1)  # word_num x op_num
        word_op = word_op.masked_fill(word_op < self.op_threshold, 0)
        return word_word, word_op
    
    def save_know(self, word_word, word_op):
        torch.save(word_word, self.ww_know_save)
        torch.save(word_op, self.wo_know_save)
    
    def recall_know(self, delta_word_word, delta_word_op):
        self.delta_word_word = delta_word_word
        self.delta_word_op = delta_word_op
        return 

    def forward(self, word_output):
        # word_output: B x S x 768
        dis = -torch.norm(self.linear(word_output).unsqueeze(-2)-self.op_emb, p=2, dim=-1) 
        prob = torch.softmax(dis, dim=-1)
        prob = prob.masked_fill(prob < self.op_threshold, 0)
        return prob
        
class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(
            torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums, hidden_size * 2), a=-1/(hidden_size * 2), b=1/(hidden_size * 2)))
        self.ops_bias = nn.Parameter(torch.nn.init.uniform_(torch.empty(op_nums), a=-1/(hidden_size * 2), b=1/(hidden_size * 2)))

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)
        
        self.hgcn = HGCN(hidden_size, hidden_size, op_nums, dropout)

    def verse_forward(self, c, r):
        ld = self.dropout(r)
        c = self.dropout(c)
        g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
        t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
        return g * t
        
    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, word_word, word_op, word_exist_mat, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(
            0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(
            encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(
            *repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat(
            (embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1),
                               embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        now_ops, word_op = self.hgcn(encoder_outputs.transpose(0,1), self.ops, current_embeddings, current_attn, word_word, word_op, word_exist_mat, seq_mask)
        op = (leaf_input.unsqueeze(1) * now_ops).sum(-1) + self.ops_bias

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight, word_op


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(
            hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(
            torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(
            torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(
            torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(
            torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(
            torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(
            torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree

class HGCN(nn.Module):
    def __init__(self, input_dim, hidden_size, op_nums, ft=0.99, dropout=0.5):
        super(HGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.ww_gcn = GCN(hidden_size * 2, hidden_size, hidden_size * 2, dropout)
        self.norm = LayerNorm1(hidden_size * 2)
        self.norm1 = LayerNorm1(hidden_size)
        self.norm2 = LayerNorm1(hidden_size * 2)
        self.w_o = nn.Linear(hidden_size * 2, hidden_size)
        # self.w_o = nn.Linear(hidden_size, hidden_size)
        self.o_trans = nn.Linear(hidden_size * 3, hidden_size * 2)
        # self.o_output = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = dropout
        self.ft = ft
        
        self.w_o_forget = nn.Linear(hidden_size * 2 + op_nums * hidden_size * 2, op_nums)
        self.w_o_rel = nn.Linear(hidden_size * 2+op_nums*hidden_size * 2, op_nums)
        self.w_o_update = nn.Linear(2*op_nums, op_nums)
    
    def normalize_matrix(self, matrix):
        diag = torch.sum(matrix, dim=-1, keepdims=True)
        return matrix / (diag+1e-30)

    def update_graph(self, word_outputs, ops, word_operator_old):
        # update word-operator       
        word_output = word_outputs / (torch.norm(word_outputs, p=2,dim=-1,keepdim=True) + 1e-30)
        op = ops / (torch.norm(ops, p=2,dim=-1,keepdim=True) + 1e-30)
        op_reshape = op.reshape(op.size(0),-1).unsqueeze(1).repeat(1,word_output.size(1),1)
        concate = torch.cat([word_output, op_reshape],dim=-1)
        forget = torch.clip(torch.sigmoid(self.w_o_forget(concate)),min=self.ft)
        updated = F.softmax(F.relu(self.w_o_update(torch.cat([word_operator_old, self.w_o_rel(concate)],dim=-1))),dim=-1)
        word_operator = forget * word_operator_old + (1-forget) * updated
        return word_operator
        
    def forward(self, encoder_outputs, ops, s, current_attn, word_word, word_op, word_exist_mat, seq_mask):                    
        # encoder_outputs: B x seq x N, ops: op_nums x 2N
        # s: B x 1 x N
        # current_attn: B x 1 x seq
        # word_word: B x seq x seq, word_op: B x seq x op x 2
        # word_exist_mat: B x seq x seq, seq_mask: B x seq
        
        batch_size = encoder_outputs.size(0)
        # s -> word
        s2w = current_attn.transpose(1,2) * s  # B x seq x N
        w_all = torch.cat([s2w, encoder_outputs], dim=-1)  
        ww_adj = word_word.squeeze(-1).masked_fill(word_exist_mat!=1, 0)
        w2w = self.ww_gcn(w_all, ww_adj)
        w2w = self.norm(w2w) + w_all  # B x seq x 2N
        # w2w = encoder_outputs
        
        # word -> operator
        wo_adj = word_op.squeeze(-1).transpose(1,2).masked_fill(seq_mask.unsqueeze(1).bool(), 0)  # B x op x seq 
        op_trans = F.relu(self.w_o(torch.matmul(self.normalize_matrix(wo_adj), w2w)))
        op_trans = F.dropout(op_trans, self.dropout, training=self.training)  # B x op x N
        op_trans = self.norm1(op_trans)

        op_all = torch.cat([op_trans, torch.unsqueeze(ops, 0).repeat(batch_size, 1, 1)], dim=-1)  # B x op x 3N
        op_h = F.relu(self.o_trans(op_all))
        op_o = self.norm2(op_h) + ops
        
        wo = self.update_graph(w2w, op_o, word_op)
        return op_o, wo

# Graph Module
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class LayerNorm1(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm1, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(torch.sum((x-mean)**2,dim=-1,keepdim=True))
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(Graph_Module, self).__init__()
        '''
        ## Variables:
        - indim: dimensionality of input node features
        - hiddim: dimensionality of the joint hidden embedding
        - outdim: dimensionality of the output node features
        - combined_feature_dim: dimensionality of the joint hidden embedding for graph
        - K: number of graph nodes/objects on the image
        '''
        self.in_dim = indim
        #self.combined_dim = outdim
        
        #self.edge_layer_1 = nn.Linear(indim, outdim)
        #self.edge_layer_2 = nn.Linear(outdim, outdim)
        
        #self.dropout = nn.Dropout(p=dropout)
        #self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        #self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)
        self.h = 4
        self.d_k = outdim//self.h
        
        #layer = GCN(indim, hiddim, self.d_k, dropout)
        self.graph = clones(GCN(indim, hiddim, self.d_k, dropout), 4)
        
        #self.Graph_0 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_1 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_2 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_3 = GCN(indim, hiddim, outdim//4, dropout)
        
        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        self.K = graph_nodes.size(1)
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)
        
        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)
        
        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        
        adjacency_matrix = self.b_normal(adjacency_matrix)

        return adjacency_matrix
    
    def normalize(self, A, symmetric=True):
        '''
        ## Inputs:
        - adjacency matrix (K, K) : A
        ## Returns:
        - adjacency matrix (K, K) 
        '''
        A = A + torch.eye(A.size(0)).cuda().float()
        d = A.sum(1)
        if symmetric:
            # D = D^{-1/2}
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(A)
       
    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - graph_encode_features (batch_size, K, out_feat_dim)
        '''
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        # adj (batch_size, K, K): adjacency matrix
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            #adj = adj.unsqueeze(1)
            #adj = torch.cat((adj,adj,adj),1)
            adj_list = [adj,adj,adj,adj]
        else:
            adj = graph.float()
            adj_list = [adj[:,1,:],adj[:,1,:],adj[:,4,:],adj[:,4,:]]
        #print(adj)
        
        g_feature = \
            tuple([l(graph_nodes,x) for l, x in zip(self.graph,adj_list)])
        #g_feature_0 = self.Graph_0(graph_nodes,adj[0])
        #g_feature_1 = self.Graph_1(graph_nodes,adj[1])
        #g_feature_2 = self.Graph_2(graph_nodes,adj[2])
        #g_feature_3 = self.Graph_3(graph_nodes,adj[3])
        #print('g_feature')
        #print(type(g_feature))
        
        
        g_feature = self.norm(torch.cat(g_feature,2)) + graph_nodes
        #print('g_feature')
        #print(g_feature.shape)
        
        graph_encode_features = self.feed_foward(g_feature) + g_feature
        
        return adj, graph_encode_features

# GCN
class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
# Graph_Conv
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.shape)
        #print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        #print(adj.shape)
        #print(support.shape)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'