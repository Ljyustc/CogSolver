# -*- coding: utf-8 -*-
import os
from src.config import args
import time
import logging
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
import torch
USE_CUDA = torch.cuda.is_available()

from torch.autograd import grad
from src.masked_cross_entropy import *
from src.train_and_evaluate import *

PAD_token = 0
def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq[:max_length]
    
def cal_loss(data_select, models, meta_term, get_know=False, ww=None, wo=None):
    input_batch, input_length, target_batch, target_length, _, nums_stack_batch, num_pos, num_size_batch = data_select
    embedding, encoder, predict, generate, merge, knowledge_base = models
    generate_nums, input_lang, output_lang = meta_term
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.BoolTensor(seq_mask)
    
    input_batch_new = []
    for i in input_batch:
        input_batch_new.append(pad_seq(i, len(i), max_len))
    
    target_max_len = max(target_length)
    target_batch_new = []
    for i in target_batch:
        target_batch_new.append(pad_seq(i, len(i), target_max_len))
    
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["UNK"]
    
    # print("input_length", input_length)
    # print("input_batch", input_batch_new)
    input_var = torch.LongTensor(input_batch_new).transpose(0, 1)
    target = torch.LongTensor(target_batch_new).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)
    
    embedding.eval()
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    knowledge_base.eval()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    
    embedded = embedding(input_var)
    # Run words through encoder
    word_exist_mat = word_exist(seq_mask)
    
    encoder_outputs, problem_output, word_word, word_op = encoder(embedded, input_length, knowledge_base)
    if get_know:
        word_word, word_op = knowledge_base.get_know_from(input_batch_new, ww, wo)
    else:
        delta_word_word, delta_word_operator = knowledge_base.get_delta_know(input_batch_new)
        word_word += delta_word_word
        word_op += delta_word_operator

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    verse_node_stacks, verse_left, verse_goal, cur = [[] for _ in range(batch_size)], [], [], []

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    comm = output_lang.comm
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings, word_op = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, word_word, word_op, word_exist_mat, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk, max_tar=outputs.size(-1))
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o, c_g in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks, current_embeddings):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                if i in comm:
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False, True))
                    verse_node_stacks[idx].append(l)
                else:
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False, False))
            else:
                if i - num_start >= current_nums_embeddings.size(1):
                    po_ = min(num_start, current_nums_embeddings.size(1)-1)
                else:
                    po_ = i - num_start
                current_num = current_nums_embeddings[idx, po_].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)
            # if len(o) > 1 and o[-2].commutative:
                # verse_node_stacks[idx].append(c_g)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    loss = masked_cross_entropy(all_node_outputs, target, target_length)
        
    return loss

def grad_z(batch_data, models, meta_term, get_know=False, ww=None, wo=None, create_graph=False, enabled=True, allow_unused=False):
    with torch.backends.cudnn.flags(enabled=enabled):
        loss = cal_loss(batch_data, models, meta_term, get_know=get_know, ww=ww, wo=wo)
    # Compute sum of gradients from model parameters to loss
    params = []
    for model in models:
       params += [ p for p in model.parameters() if p.requires_grad ]
    return list(grad(loss, params, create_graph=create_graph, allow_unused=allow_unused))

def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    return_grads = grad(y, w, allow_unused=True)
    # First backprop
    # first_grads = grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)

    # # Elementwise products
    # elemwise_products = 0
    # for grad_elem, v_elem in zip(first_grads, v):
        # if grad_elem is not None and v_elem is not None:
            # elemwise_products += torch.sum(grad_elem * v_elem)

    # # Second backprop
    # return_grads = grad(elemwise_products, w, allow_unused=True)
    return [-x if x is not None else 0 for x in return_grads]
    
def s_test(data_reh, train_unzip, models, meta_term, get_know=False, ww=None, wo=None, damp=0.01, scale=25.0, recursion_depth=500):
    v = grad_z(data_reh, models, meta_term, enabled=False, allow_unused=True)
    h_estimate = v.copy()
    params = []
    for model in models:
        params += [ p for p in model.parameters() if p.requires_grad ]
    if len(h_estimate) != len(params):
        return False
    for i in range(recursion_depth):
        id_ = random.choice(range(len(train_unzip)))  # random choice a train sample for strochastic estimation
        data_select = train_unzip[id_]
        with torch.backends.cudnn.flags(enabled=False):
            loss = cal_loss(data_select, models, meta_term, get_know=get_know, ww=ww, wo=wo)
        hv = hvp(loss, params, h_estimate)
        # Recursively caclulate h_estimate
        h_estimate = [_v + (1 - damp) * _h_e - _hv / scale for _v, _h_e, _hv in zip(v, h_estimate, hv) if (_h_e!=None and v!=None)]
        if len(h_estimate) != len(params):
            break
    return h_estimate

def influence_compute(data_reh, word_word_old, word_op_old, models, meta_term, grad_z_vec, s_test_vec, train_dataset_size):
    loss = cal_loss(data_reh, models, meta_term, get_know=True, ww=word_word_old, wo=word_op_old)
    grad_know_1 = grad(loss, [word_word_old, word_op_old])
    grad_know_2 = grad(sum([torch.sum(k * j) for k, j in zip(grad_z_vec, s_test_vec) if k!=None]) / train_dataset_size, [word_word_old, word_op_old])
    
    influence_ww = grad_know_1[0] - grad_know_2[0]
    influence_wo = grad_know_1[1] - grad_know_2[1]
    return influence_ww, influence_wo

def rehearsal_know(data_train, data_reh, word_word_old, word_op_old, models, meta_term, recursion_depth, r):
    # unzip train data for efficient sampling
    train_unzip = []
    train_dataset_size = len(data_train[0])
    for id_ in range(train_dataset_size):
        new_input = data_train[0][id_][:data_train[1][id_]]
        new_output = data_train[2][id_][:data_train[3][id_]]
        a = [[new_input]] + [[x[id_]] for x in data_train[1:]]
        a[2] = [new_output]
        train_unzip.append(a)
    
    s_test_vec_list = []
    for i in range(r):
        s_test_result = s_test(data_reh, train_unzip, models, meta_term, get_know=True, ww=word_word_old, wo=word_op_old, recursion_depth=recursion_depth)
        if s_test_result:
            s_test_vec_list.append(s_test_result)

    s_test_vec = s_test_vec_list[0]
    for i in range(r):
        s_test_vec = [s_test_vec[x] + s_test_vec_list[i][x] for x in range(len(s_test_vec))]

    s_test_vec = [(i / r).detach() for i in s_test_vec]  # avoid computing gradient for knowledge
    
    # Calculate the influence function
    grad_z_vec = grad_z(data_train, models, meta_term, get_know=True, ww=word_word_old, wo=word_op_old, create_graph=True, enabled=False, allow_unused=True)
    
    influence_ww, influence_wo = influence_compute(data_reh, word_word_old, word_op_old, models, meta_term, grad_z_vec, s_test_vec, train_dataset_size)
    
    # gradient for knowledge may vanish, resulting in influence being Nan
    influence_ww = torch.where(torch.isnan(influence_ww), torch.full_like(influence_ww, 0), influence_ww)
    influence_wo = torch.where(torch.isnan(influence_wo), torch.full_like(influence_wo, 0), influence_wo)
    return influence_ww, influence_wo