# -*- coding: utf-8 -*-

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from .attention import get_mask, HierarchicalAttention

class Bottom2Up_Net(nn.Module):
    def __init__(self, input_dim, hidden_size, ft, w_w_s, w_w_k, w_update):
        super(Bottom2Up_Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.ft = ft
        
        self.g_w = nn.Linear(input_dim, hidden_size)
        self.w_w_s = w_w_s
        self.w_w_k = w_w_k
        self.w_o = nn.Linear(input_dim, input_dim)
        self.w_update = w_update
        self.forget_gate = nn.Linear(hidden_size*3 + input_dim, input_dim)
        self.forget_gate2 = nn.Linear(input_dim*2, input_dim)
        self.linear_output = nn.Linear(input_dim, input_dim)
    
    def normalize_matrix(self, matrix):
        return matrix / ((matrix!=0).sum(-1,keepdim=True)+1e-30)

    def forward(self, input_embeddings, word_exist_sequence, word_exist_matrix, graph_adjs, goal_word):
        word_outputs, node_hidden, op_embedding = input_embeddings
        all_word_embedding = torch.cat(word_outputs, dim=1)  # batch_size, all_words, input_dim
        word_operator, word_word, depend_relation = graph_adjs
        words_length = [span.size(1) for span in depend_relation]
        # goal -> word
        all_words = goal_word.size(1)
        goal_trans = self.g_w(node_hidden).unsqueeze(1).expand(-1, all_words, -1)
        word_g = goal_trans * goal_word.unsqueeze(-1)  # batch_size, all_word, hidden_size
        word_g = word_g / (torch.norm(word_g,p=2,dim=-1,keepdim=True) + 1e-30)
        
        # word -- word
        word_know_trans = self.w_w_k(all_word_embedding)
        word_w_k = torch.matmul(self.normalize_matrix(word_word * word_exist_matrix), word_know_trans)  # batch_size, all_word, hidden_size
        word_w_k = word_w_k / (torch.norm(word_w_k,p=2,dim=-1,keepdim=True) + 1e-30)
        
        # dependency
        word_w_s = [torch.matmul(self.normalize_matrix(depend_relation[i]), self.w_w_s(word_outputs[i])) for i in range(len(word_outputs))]
        word_w_s = torch.cat(word_w_s, dim=1)  # batch_size, all_word, hidden_size
        word_w_s = word_w_s / (torch.norm(word_w_s,p=2,dim=-1,keepdim=True) + 1e-30)
                
        word_neighbor = torch.cat([word_g, word_w_k, word_w_s], dim=-1)
        forget = torch.sigmoid(self.forget_gate(torch.cat([all_word_embedding, word_g, word_w_k, word_w_s], dim=-1)))
        word_updated = torch.clip(forget,min=self.ft) * all_word_embedding + (1-forget) * F.relu(self.w_update(word_neighbor))  # batch_size, all_word, input_dim
        
        # word -> operator
        word_op = torch.matmul(self.normalize_matrix(torch.transpose(word_exist_sequence.unsqueeze(-1) * word_operator, 1,2)), self.w_o(word_updated)) # batch_size, operation, input_dim
        forget2 = torch.sigmoid(self.forget_gate2(torch.cat([op_embedding, word_op], dim=-1)))
        op_embedding = torch.clip(forget2,min=self.ft) * op_embedding + (1-forget2) * F.relu(self.linear_output(word_op))
        
        word_outputs = torch.split(word_updated, words_length, dim=1)
        return word_outputs, op_embedding

class Up2Bottom_Net(nn.Module):
    def __init__(self, input_dim, hidden_size, ft, w_w_s, w_w_k, w_update):
        super(Up2Bottom_Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.ft = ft
        
        self.o_w = nn.Linear(input_dim, hidden_size)
        self.w_w_s = w_w_s
        self.w_w_k = w_w_k
        self.w_update = w_update
        self.w_g = nn.Linear(input_dim, input_dim)
        self.forget_gate = nn.Linear(input_dim*2, input_dim)
    
    def normalize_matrix(self, matrix):
        return matrix / ((matrix!=0).sum(-1,keepdim=True)+1e-30)

    def forward(self, input_embeddings, word_exist_matrix, word_exist_sequence, graph_adjs, goal_word):
        word_outputs, node_hidden, op_embedding = input_embeddings
        all_word_embedding = torch.cat(word_outputs, dim=1)  # batch_size, all_words, input_dim
        word_operator, word_word, depend_relation = graph_adjs
        
        # operator -> word
        word_o = torch.matmul(word_operator, self.o_w(op_embedding))  # batch_size, all_words, hidden_size
        word_o = word_o / (torch.norm(word_o,p=2,dim=-1,keepdim=True) + 1e-30)
        
        # word -- word
        word_know_trans = self.w_w_k(all_word_embedding)
        word_w_k = torch.matmul(self.normalize_matrix(word_word * word_exist_matrix), word_know_trans)  # batch_size, all_word, hidden_size
        word_w_k = word_w_k / (torch.norm(word_w_k,p=2,dim=-1,keepdim=True) + 1e-30)
        
        # dependency
        word_w_s = [torch.matmul(self.normalize_matrix(depend_relation[i]), self.w_w_s(word_outputs[i])) for i in range(len(word_outputs))]
        word_w_s = torch.cat(word_w_s, dim=1)  # batch_size, all_word, hidden_size
        word_w_s = word_w_s / (torch.norm(word_w_s,p=2,dim=-1,keepdim=True) + 1e-30)
        
        word_neighbor = torch.cat([word_o, word_w_k, word_w_s], dim=-1)
        word_updated = F.relu(self.w_update(word_neighbor))  # batch_size, all_word, input_dim
        
        # word -> goal 
        goal_w = torch.bmm(self.normalize_matrix(goal_word * word_exist_sequence).unsqueeze(1), word_updated).squeeze(1)
        goal_updated = F.relu(self.w_g(goal_w))  # batch_size, input_dim
        forget = torch.sigmoid(self.forget_gate(torch.cat([goal_w, node_hidden], dim=-1)))
        goal_next = torch.clip(forget,min=self.ft) * node_hidden + (1-forget) * goal_updated
        return goal_next
        
class HGCN(nn.Module):
    def __init__(self, input_dim, hidden_size, op_num, ft):
        super(HGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.ft = ft
        
        # share parameters
        self.w_w_s = nn.Linear(input_dim, hidden_size)
        self.w_w_k = nn.Linear(input_dim, hidden_size)
        self.w_update = nn.Linear(hidden_size*3, input_dim)
        
        self.bottom2up = Bottom2Up_Net(input_dim, hidden_size, self.ft, self.w_w_s, self.w_w_k, self.w_update)
        self.up2bottom = Up2Bottom_Net(input_dim, hidden_size, self.ft, self.w_w_s, self.w_w_k, self.w_update)
        
        self.w_o_forget = nn.Linear(input_dim + op_num * input_dim, op_num)
        self.w_o_rel = nn.Linear(input_dim+op_num*input_dim, op_num)
        self.w_o_update = nn.Linear(2*op_num, op_num)

    def update_graph(self, word_outputs, op_embedding, word_operator_old):
        # update word-operator       
        all_word_embedding = torch.cat(word_outputs, dim=1)
        op = op_embedding
        all_word_embedding = all_word_embedding / (torch.norm(all_word_embedding, p=2,dim=-1,keepdim=True) + 1e-30)
        op = op / (torch.norm(op, p=2,dim=-1,keepdim=True) + 1e-30)
        op_reshape = op.reshape(op.size(0),-1).unsqueeze(1).repeat(1,all_word_embedding.size(1),1)
        concate = torch.cat([all_word_embedding, op_reshape],dim=-1)
        forget = torch.clip(torch.sigmoid(self.w_o_forget(concate)),min=self.ft)
        updated = F.softmax(F.relu(self.w_o_update(torch.cat([word_operator_old, self.w_o_rel(concate)],dim=-1))),dim=-1)
        word_operator = forget * word_operator_old + (1-forget) * updated
        return word_operator
        
    def forward(self, input_embeddings, word_exist_sequence, word_exist_matrix, graph_adjs, goal_word):
        # bottom -> up
        word_outputs, op_embedding = self.bottom2up(input_embeddings, word_exist_sequence, word_exist_matrix, graph_adjs, goal_word)

        # update relation
        graph_adjs[0] = self.update_graph(word_outputs, op_embedding, graph_adjs[0])
        
        # up -> bottom
        input_embeddings = (word_outputs, input_embeddings[1], op_embedding)
        goal_next_temp = self.up2bottom(input_embeddings, word_exist_matrix, word_exist_sequence, graph_adjs, goal_word)
        return word_outputs, op_embedding, goal_next_temp

class GateNN(nn.Module):
    def __init__(self, hidden_size, input1_size, input2_size=0, dropout=0.4, single_layer=False):
        super(GateNN, self).__init__()
        self.single_layer = single_layer
        self.hidden_l1 = nn.Linear(input1_size+hidden_size, hidden_size)
        self.gate_l1 = nn.Linear(input1_size+hidden_size, hidden_size)
        if not single_layer:
            self.dropout = nn.Dropout(p=dropout)
            self.hidden_l2 = nn.Linear(input2_size+hidden_size, hidden_size)
            self.gate_l2 = nn.Linear(input2_size+hidden_size, hidden_size)
        return
    
    def forward(self, hidden, input1, input2=None):
        input1 = torch.cat((hidden, input1), dim=-1)
        h = torch.tanh(self.hidden_l1(input1))
        g = torch.sigmoid(self.gate_l1(input1))
        h = h * g
        if not self.single_layer:
            h1 = self.dropout(h)
            if input2 is not None:
                input2 = torch.cat((h1, input2), dim=-1)
            else:
                input2 = h1
            h = torch.tanh(self.hidden_l2(input2))
            g = torch.sigmoid(self.gate_l2(input2))
            h = h * g
        return h

class ScoreModel(nn.Module):
    def __init__(self, hidden_size):
        super(ScoreModel, self).__init__()
        self.w = nn.Linear(hidden_size * 3, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        return
    
    def forward(self, hidden, context, token_embeddings):
        # hidden/context: batch_size * hidden_size
        # token_embeddings: batch_size * class_size * hidden_size
        batch_size, class_size, _ = token_embeddings.size()
        hc = torch.cat((hidden, context), dim=-1)
        # (b, c, h)
        hc = hc.unsqueeze(1).expand(-1, class_size, -1)
        hidden = torch.cat((hc, token_embeddings), dim=-1)
        hidden = F.leaky_relu(self.w(hidden))
        score = self.score(hidden).view(batch_size, class_size)
        return score

class PredictModel(nn.Module):
    def __init__(self, hidden_size, class_size, op_num, ft, dropout=0.4, use_cuda=True):
        super(PredictModel, self).__init__()
        self.class_size = class_size

        self.dropout = nn.Dropout(p=dropout)
        self.use_cuda = use_cuda
        self.attn = HierarchicalAttention(hidden_size)
        
        self.score_pointer = ScoreModel(hidden_size)
        self.score_generator = ScoreModel(hidden_size)
        self.score_span = ScoreModel(hidden_size)
        self.gen_prob = nn.Linear(hidden_size*2, 1)
        self.op_prob = nn.Linear(hidden_size*2, 1)
        self.hgcn = HGCN(hidden_size, hidden_size, op_num, ft)
        return
    
    def get_pointer_embedding(self, pointer_num_pos, encoder_outputs):
        # encoder_outputs: batch_size * seq_len * hidden_size
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        batch_size, pointer_size = pointer_num_pos.size()
        batch_index = torch.arange(batch_size)
        if self.use_cuda:
            batch_index = batch_index.cuda()
        batch_index = batch_index.unsqueeze(1).expand(-1, pointer_size)
        # batch_size * pointer_len * hidden_size
        pointer_embedding = encoder_outputs[batch_index, pointer_num_pos]
        # mask invalid pos -1
        pointer_embedding = pointer_embedding * (pointer_num_pos != -1).unsqueeze(-1)
        return pointer_embedding
    
    def score_pn(self, hidden, context, embedding_masks, goal_word):
        
        # embedding: batch_size * pointer_size * hidden_size
        # mask: batch_size * pointer_size
        pointer_mask, pointer_embedding, generator_embedding, _ = embedding_masks
        hidden = self.dropout(hidden)
        context = self.dropout(context)
        pointer_embedding = self.dropout(pointer_embedding)
        pointer_score = self.score_pointer(hidden, context, pointer_embedding)
        pointer_score = pointer_score.masked_fill(pointer_mask, -float('inf'))
        # batch_size * symbol_size
        # pointer
        pointer_prob = F.softmax(pointer_score, dim=-1)
        
        generator_embedding = self.dropout(generator_embedding)
        generator_score = self.score_generator(hidden, context, generator_embedding)
        # batch_size * generator_size
        # generator
        generator_prob = F.softmax(generator_score, dim=-1)
        return pointer_prob, generator_prob

    def forward(self, node_hidden, encoder_outputs, graph_adjs, masks, meta_value):
        use_cuda = node_hidden.is_cuda
        node_hidden_dropout = self.dropout(node_hidden).unsqueeze(1)
        span_output, word_outputs = encoder_outputs
        span_mask, word_masks, word_exist_sequence, word_exist_matrix = masks
        output_attn, goal_word = self.attn(node_hidden_dropout, span_output, word_outputs, span_mask, word_masks)
        context = output_attn.squeeze(1)
        
        ######
        pointer_mask, word_pointer_num_poses, non_op_generator_embedding, op_embedding = meta_value
        input_embeddings = (word_outputs, node_hidden, op_embedding)
        word_outputs, op_embedding, goal_next = self.hgcn(input_embeddings, word_exist_sequence, word_exist_matrix, graph_adjs, goal_word)
        encoder_outputs[-1] = word_outputs
        meta_value[-1] = op_embedding
        ######
        hc = torch.cat((node_hidden, context), dim=-1)
        # log(f(softmax(x)))
        # prob: softmax
        num_pointer_embeddings = []
        for word_output, word_pointer_num_pos in zip(word_outputs, word_pointer_num_poses):
            num_pointer_embedding = self.get_pointer_embedding(word_pointer_num_pos, word_output)
            num_pointer_embeddings.append(num_pointer_embedding)
        pointer_embedding = torch.cat([embedding.unsqueeze(0) for embedding in num_pointer_embeddings], dim=0).sum(dim=0)
        generator_embedding = torch.cat((op_embedding, non_op_generator_embedding), dim=1)
        all_embedding = torch.cat((generator_embedding, pointer_embedding), dim=1)
        embedding_masks = (pointer_mask, pointer_embedding, generator_embedding, all_embedding)
        
        pointer_prob, generator_prob = self.score_pn(node_hidden, context, embedding_masks, goal_word)
        gen_prob = torch.sigmoid(self.gen_prob(hc))
        
        prob = torch.cat((gen_prob * generator_prob, (1 - gen_prob) * pointer_prob), dim=-1)
        # batch_size * class_size
        # operation + generator + pointer + empty_pointer
        pad_empty_pointer = torch.zeros(prob.size(0), self.class_size - prob.size(-1))
        if use_cuda:
            pad_empty_pointer = pad_empty_pointer.cuda()
        prob = torch.cat((prob, pad_empty_pointer), dim=-1)
        output = torch.log(prob + 1e-30)
        return output, context, embedding_masks, goal_next

class TreeEmbeddingNode:
    def __init__(self, embedding, terminal, commutative):
        self.embedding = embedding
        self.terminal = terminal
        self.commutative = commutative
        return

class TreeEmbeddingModel(nn.Module):
    def __init__(self, hidden_size, op_set, commutative_op_set, dropout=0.4):
        super(TreeEmbeddingModel, self).__init__()
        self.op_set = op_set
        self.commutative_op_set = commutative_op_set
        self.dropout = nn.Dropout(p=dropout)
        self.combine = GateNN(hidden_size, hidden_size*2, dropout=dropout, single_layer=True)
        return
    
    def merge(self, op_embedding, left_embedding, right_embedding):
        te_input = torch.cat((left_embedding, right_embedding), dim=-1)
        te_input = self.dropout(te_input)
        op_embedding = self.dropout(op_embedding)
        tree_embed = self.combine(op_embedding, te_input)
        return tree_embed
    
    def forward(self, class_embedding, tree_stacks, verse_node_stacks, embed_node_index, decompose=None):
        # embed_node_index: batch_size
        use_cuda = embed_node_index.is_cuda
        batch_index = torch.arange(embed_node_index.size(0))
        if use_cuda:
            batch_index = batch_index.cuda()
        labels_embedding = class_embedding[batch_index, embed_node_index]
        for idx, node_label, tree_stack, label_embedding in zip(range(len(tree_stacks)), embed_node_index.cpu().tolist(), tree_stacks, labels_embedding):
            # operations
            if node_label in self.op_set:
                if node_label in self.commutative_op_set:
                    tree_node = TreeEmbeddingNode(label_embedding, terminal=False, commutative=True)
                else:
                    tree_node = TreeEmbeddingNode(label_embedding, terminal=False, commutative=False)
            # numbers
            else:
                right_embedding = label_embedding
                # on right tree => merge
                while len(tree_stack) >= 2 and tree_stack[-1].terminal and (not tree_stack[-2].terminal):
                    if tree_stack[-2].commutative and decompose is not None:
                        left_node = verse_node_stacks[0][idx].pop()
                        parent_node = verse_node_stacks[0][idx].pop()
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        r_input = torch.cat((node_context, label_embedding), dim=-1)
                        # r_input = self.dropout(r_input)
                        # node_hidden = self.dropout(node_hidden)
                        verse_left_hidden = decompose(node_hidden, r_input, right_embedding)
                        verse_node_stacks[1].append(torch.norm(left_node.node_hidden - verse_left_hidden))
                    left_embedding = tree_stack.pop().embedding
                    op_embedding = tree_stack.pop().embedding
                    right_embedding = self.merge(op_embedding, left_embedding, right_embedding)
                tree_node = TreeEmbeddingNode(right_embedding, terminal=True, commutative=False)
            tree_stack.append(tree_node)
        return labels_embedding

class NodeEmbeddingNode:
    def __init__(self, node_hidden, node_context=None, label_embedding=None):
        self.node_hidden = node_hidden
        self.node_context = node_context
        self.label_embedding = label_embedding
        return

class DecomposeModel(nn.Module):
    def __init__(self, hidden_size, dropout=0.4, use_cuda=True):
        super(DecomposeModel, self).__init__()
        self.pad_hidden = torch.zeros(hidden_size)
        if use_cuda:
            self.pad_hidden = self.pad_hidden.cuda()

        self.dropout = nn.Dropout(p=dropout)
        self.l_decompose = GateNN(hidden_size, hidden_size*2, 0, dropout=dropout, single_layer=False)
        self.r_decompose = GateNN(hidden_size, hidden_size*2, hidden_size, dropout=dropout, single_layer=False)
        return

    def forward(self, node_stacks, tree_stacks, verse_node_stacks, goals_next, nodes_context, labels_embedding, pad_node=True):
        children_hidden = []
        for idx, node_stack, tree_stack, goal_next, node_context, label_embedding in zip(range(len(tree_stacks)), node_stacks, tree_stacks, goals_next, nodes_context, labels_embedding):
            # start from encoder_hidden
            # len == 0 => finished decode
            if len(node_stack) > 0:
                # left
                if not tree_stack[-1].terminal:
                    node_hidden = goal_next    # parent, still need for right
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)   # add context and label of parent for right child                    
                    l_input = torch.cat((node_context, label_embedding), dim=-1)
                    l_input = self.dropout(l_input)
                    node_hidden = self.dropout(node_hidden)
                    child_hidden = self.l_decompose(node_hidden, l_input, None)
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for left child      

                    if verse_node_stacks is not None and tree_stack[-1].commutative:
                        verse_node_stacks[0][idx] += [node_stack[-2], NodeEmbeddingNode(child_hidden, None, None)]                    
                # right
                else:
                    node_stack.pop()    # left child, no need
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()  # parent, no longer need
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding   # left tree
                        left_embedding = self.dropout(left_embedding)
                        r_input = torch.cat((node_context, label_embedding), dim=-1)
                        r_input = self.dropout(r_input)
                        node_hidden = self.dropout(node_hidden)
                        child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
                        node_stack.append(NodeEmbeddingNode(child_hidden, None, None))  # only hidden for right child
                    # else finished decode
            # finished decode, pad
            if len(node_stack) == 0:
                child_hidden = self.pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        return children_hidden

def copy_list(src_list):
    dst_list = [copy_list(item) if type(item) is list else item for item in src_list]
    return dst_list

class BeamNode:
    def __init__(self, score, nodes_hidden, node_stacks, tree_stacks, encoder_outputs, graph_adj, meta_value, decoder_outputs_list, sequence_symbols_list):
        self.score = score
        self.nodes_hidden = nodes_hidden
        self.node_stacks = node_stacks
        self.tree_stacks = tree_stacks
        self.encoder_outputs = encoder_outputs
        self.graph_adj = graph_adj
        self.meta_value = meta_value
        self.decoder_outputs_list = decoder_outputs_list
        self.sequence_symbols_list = sequence_symbols_list
        return
    
    def copy(self):
        node = BeamNode(
            self.score,
            self.nodes_hidden,
            copy_list(self.node_stacks),
            copy_list(self.tree_stacks),
            copy_list(self.encoder_outputs),
            copy_list(self.graph_adj),
            copy_list(self.meta_value),
            copy_list(self.decoder_outputs_list),
            copy_list(self.sequence_symbols_list)
        )
        return node

class Decoder(nn.Module):
    def __init__(self, embed_model, op_set, vocab_dict, class_list, ft=0.99, hidden_size=512, dropout=0.4, use_cuda=True, knowledge_base=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.op_size = len(op_set)
        embed_size = embed_model.embedding_dim
        class_size = len(class_list)

        self.get_predict_meta(class_list, vocab_dict, op_set)

        self.embed_model = embed_model
        # 128 => 512
        self.op_hidden = nn.Linear(embed_size, hidden_size)
        self.predict = PredictModel(hidden_size, class_size, self.op_size, ft=ft, dropout=dropout, use_cuda=use_cuda)
        op_set = set(i for i, symbol in enumerate(self.embed_list) if symbol in op_set)
        commutative_op_set = set(i for i, symbol in enumerate(self.embed_list) if symbol in ['+','*'])
        self.tree_embedding = TreeEmbeddingModel(hidden_size, op_set, commutative_op_set, dropout=dropout)
        self.decompose = DecomposeModel(hidden_size, dropout=dropout, use_cuda=use_cuda)
        self.knowledge_base = knowledge_base

    def get_predict_meta(self, class_list, vocab_dict, op_set):
        # embed order: opt_generator + non_opt_generator + pointer
        # used in predict_model, tree_embedding
        pointer_list = [token for token in class_list if 'temp_' in token]
        generator_list = [token for token in class_list if token not in pointer_list and token not in op_set]
        embed_list = list(op_set) + generator_list + pointer_list

        # pointer num index in class_list, for select only num pos from num_pos with op pos
        self.pointer_index = torch.LongTensor([class_list.index(token) for token in pointer_list])
        # generator symbol index in vocab, for generator symbol embedding
        self.generator_vocab = torch.LongTensor([vocab_dict[token] for token in generator_list])  # non-operation symbol
        self.op_vocab = torch.LongTensor([vocab_dict[token] for token in list(op_set)])
        # class_index -> embed_index, for tree embedding
        # embed order -> class order, for predict_model output
        self.class_to_embed_index = torch.LongTensor([embed_list.index(token) for token in class_list])
        self.embed_list = embed_list
        if self.use_cuda:
            self.pointer_index = self.pointer_index.cuda()
            self.generator_vocab = self.generator_vocab.cuda()
            self.op_vocab = self.op_vocab.cuda()
            self.class_to_embed_index = self.class_to_embed_index.cuda()
        return

    def get_pad_masks(self, encoder_outputs, input_lengths, span_length=None):
        span_output, word_outputs = encoder_outputs
        span_pad_length = span_output.size(1)
        word_pad_lengths = [word_output.size(1) for word_output in word_outputs]
        
        span_mask = get_mask(span_length, span_pad_length)
        word_masks = [get_mask(input_length, word_pad_length) for input_length, word_pad_length in zip(input_lengths, word_pad_lengths)]
        word_exists = [(1 - word_masks[i]) * (1 - span_mask[:, [i]]) for i in range(len(word_masks))]
        
        # for all words together
        word_exist_sequence = torch.cat(word_exists, dim=-1)
        words_num = word_exist_sequence.size(-1)
        word_exist_matrix = word_exist_sequence.repeat(1, words_num).reshape(-1, words_num, words_num)
        masks = (span_mask, word_masks, word_exist_sequence, word_exist_matrix * torch.transpose(word_exist_matrix,1,2))
        return masks
    
    def get_pointer_meta(self, num_pos, sub_num_poses=None):
        batch_size = num_pos.size(0)
        pointer_num_pos = num_pos.index_select(dim=1, index=self.pointer_index)
        num_pos_occupied = pointer_num_pos.sum(dim=0) == -batch_size
        occupied_len = num_pos_occupied.size(-1)
        for i, elem in enumerate(reversed(num_pos_occupied.cpu().tolist())):
            if not elem:
                occupied_len = occupied_len - i
                break
        pointer_num_pos = pointer_num_pos[:, :occupied_len]
        # length of word_num_poses determined by span_num_pos
        if sub_num_poses is not None:
            sub_pointer_poses = [sub_num_pos.index_select(dim=1, index=self.pointer_index)[:, :occupied_len] for sub_num_pos in sub_num_poses]
        else:
            sub_pointer_poses = None
        return pointer_num_pos, sub_pointer_poses

    def get_pointer_mask(self, pointer_num_pos):
        # pointer_num_pos: batch_size * pointer_size
        # subset of num_pos, invalid pos -1
        pointer_mask = pointer_num_pos == -1
        return pointer_mask
    
    def get_generator_embedding(self, batch_size):
        # non-operation symbol generator_size * hidden_size
        generator_embedding = self.op_hidden(self.embed_model(self.generator_vocab))
        # operation generator_size * hidden_size
        op_embedding = self.op_hidden(self.embed_model(self.op_vocab))
        # batch_size * generator_size * hidden_size
        generator_embedding = generator_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        op_embedding = op_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # initialized operation embedding
        return generator_embedding, op_embedding
    
    def get_decoder_meta(self, num_pos):
        span_num_pos, word_num_poses = num_pos
        batch_size = span_num_pos.size(0)
        generator_embedding, init_op_embedding = self.get_generator_embedding(num_pos[0].size(0))
        span_pointer_num_pos, word_pointer_num_poses = self.get_pointer_meta(span_num_pos, word_num_poses)
        pointer_mask = self.get_pointer_mask(span_pointer_num_pos)
        return [pointer_mask, word_pointer_num_poses, generator_embedding, init_op_embedding]

    def init_stacks(self, encoder_hidden):
        batch_size = encoder_hidden.size(0)
        node_stacks = [[NodeEmbeddingNode(hidden, None, None)] for hidden in encoder_hidden]
        tree_stacks = [[] for _ in range(batch_size)]
        verse_node_stacks = [[[] for _ in range(batch_size)],[]]
        return node_stacks, tree_stacks, verse_node_stacks

    def forward_step(self, node_stacks, tree_stacks, verse_node_stacks, graph_adjs, nodes_hidden, encoder_outputs, masks, meta_value, decoder_nodes_class=None):
        nodes_output, nodes_context, embedding_masks, goals_next = self.predict(nodes_hidden, encoder_outputs, graph_adjs, masks, meta_value)
        nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
        predict_nodes_class = nodes_output.topk(1)[1]
        predict_nodes_prob = nodes_output.topk(1)[0]
        # teacher
        if decoder_nodes_class is not None:
            nodes_class = decoder_nodes_class.view(-1)
        # no teacher
        else:
            nodes_class = predict_nodes_class.view(-1)
        embed_nodes_index = self.class_to_embed_index[nodes_class]
        labels_embedding = self.tree_embedding(embedding_masks[-1], tree_stacks, verse_node_stacks, embed_nodes_index, self.decompose.r_decompose)
        nodes_hidden = self.decompose(node_stacks, tree_stacks, verse_node_stacks, goals_next, nodes_context, labels_embedding)
        return nodes_output, predict_nodes_class, nodes_hidden
    
    def forward_teacher(self, decoder_nodes_label, graph_adjs, decoder_init_hidden, encoder_outputs, masks, meta_value, max_length=None):
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        node_stacks, tree_stacks, verse_node_stacks = self.init_stacks(decoder_init_hidden)
        if decoder_nodes_label is not None:
            seq_len = decoder_nodes_label.size(1)
        else:
            seq_len = max_length
        for di in range(seq_len):
            if decoder_nodes_label is not None:
                decoder_node_class = decoder_nodes_label[:, di]
            else:
                decoder_node_class = None
            decoder_output, symbols, decoder_hidden = self.forward_step(node_stacks, tree_stacks, verse_node_stacks, graph_adjs, decoder_hidden, encoder_outputs, masks, meta_value, decoder_nodes_class=decoder_node_class)
            decoder_outputs_list.append(decoder_output)
            sequence_symbols_list.append(symbols)
        if decoder_nodes_label is not None:    
            return decoder_outputs_list, decoder_hidden, sequence_symbols_list, verse_node_stacks[1]
        else:
            return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def forward_beam(self, graph_adjs, decoder_init_hidden, encoder_outputs, masks, meta_value, max_length, beam_width=1):
        batch_size = decoder_init_hidden.size(0)
        node_stacks, tree_stacks, _ = self.init_stacks(decoder_init_hidden)

        word_masks = masks[1]
        word_mask = [[word_masks[span][idx].unsqueeze(0) for span in range(len(word_masks))] for idx in range(batch_size)]
        mask = [[masks[0][idx].unsqueeze(0),word_mask[idx],masks[2][idx].unsqueeze(0),masks[3][idx].unsqueeze(0)] for idx in range(batch_size)]

        span_outputs, word_outputs = encoder_outputs
        word_output = [[word_outputs[span][idx].unsqueeze(0) for span in range(len(word_outputs))] for idx in range(batch_size)]
        encoder_output = [[span_outputs[idx].unsqueeze(0), word_output[idx]] for idx in range(batch_size)]
        
        depend_relations = graph_adjs[2]
        depend_relation = [[depend_relations[span][idx].unsqueeze(0) for span in range(len(depend_relations))] for idx in range(batch_size)]
        graph_adj = [[graph_adjs[0][idx].unsqueeze(0),graph_adjs[1][idx].unsqueeze(0),depend_relation[idx]] for idx in range(batch_size)]
        
        word_pointer_num_poses = meta_value[1]
        word_pointer_num_pos = [[word_pointer_num_poses[span][idx].unsqueeze(0) for span in range(len(word_pointer_num_poses))] for idx in range(batch_size)]
        meta_values = [[meta_value[0][idx].unsqueeze(0),word_pointer_num_pos[idx], meta_value[2][idx].unsqueeze(0),meta_value[3][idx].unsqueeze(0)] for idx in range(batch_size)]
        
        decoder_outputs_list, sequence_symbols_list = [[] for _ in range(max_length)], [[] for _ in range(max_length)]
        beams_all = [[BeamNode(0, decoder_init_hidden[idx].unsqueeze(0), [node_stacks[idx]], [tree_stacks[idx]], encoder_output[idx], graph_adj[idx] , meta_values[idx], [], [])] for idx in range(batch_size)]
        for idx in range(batch_size):
            beams = beams_all[idx]
            mask_i = mask[idx]
            for _ in range(max_length):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    # finished stack-guided decoding
                    if len(b.node_stacks) == 0:
                        current_beams.append(b)
                        continue
                    # unfinished decoding
                    # batch_size * class_size
                    nodes_output, nodes_context, embedding_masks, goals_next = self.predict(b.nodes_hidden, b.encoder_outputs, b.graph_adj, mask_i, b.meta_value)
                    nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
                    # batch_size * beam_width
                    top_value, top_index = nodes_output.topk(beam_width)
                    for predict_score, predicted_symbol in zip(top_value.split(1, dim=-1), top_index.split(1, dim=-1)):
                        nb = b.copy()
                        embed_nodes_index = self.class_to_embed_index[predicted_symbol.view(-1)]
                        labels_embedding = self.tree_embedding(embedding_masks[-1], nb.tree_stacks, None, embed_nodes_index, None)
                        nodes_hidden = self.decompose(nb.node_stacks, nb.tree_stacks, None, goals_next, nodes_context, labels_embedding, pad_node=True)
                        nb.score = b.score + predict_score.item()
                        nb.nodes_hidden = nodes_hidden
                        nb.decoder_outputs_list.append(nodes_output)
                        nb.sequence_symbols_list.append(predicted_symbol)
                        current_beams.append(nb)
                beams = sorted(current_beams, key=lambda b:b.score, reverse=True)
                beams = beams[:beam_width]
                all_finished = True
                for b in beams:
                    if len(b.node_stacks[0]) != 0:
                        all_finished = False
                        break
                if all_finished:
                    break
            output = beams[0]
            for i in range(max_length):
                decoder_outputs_list[i].append(output.decoder_outputs_list[i])
                sequence_symbols_list[i].append(output.sequence_symbols_list[i])
        decoder_outputs_list = [torch.cat(decoder_outputs_list[i], dim=0) for i in range(max_length)]
        sequence_symbols_list = [torch.cat(sequence_symbols_list[i], dim=0) for i in range(max_length)]
        return decoder_outputs_list, None, sequence_symbols_list

    def forward(self, targets=None, graph_adjs=None, encoder_hidden=None, encoder_outputs=None, input_lengths=None, span_length=None, num_pos=None, max_length=None, beam_width=None):
        masks = self.get_pad_masks(encoder_outputs, input_lengths, span_length)
        meta_value = self.get_decoder_meta(num_pos)

        if type(encoder_hidden) is tuple:
            encoder_hidden = encoder_hidden[0]
        decoder_init_hidden = encoder_hidden[-1,:,:]

        if max_length is None:
            if targets is not None:
                max_length = targets.size(1)
            else:
                max_length = 40
        
        if beam_width is not None:
            return self.forward_beam(graph_adjs, decoder_init_hidden, encoder_outputs, masks, meta_value, max_length, beam_width)
        else:
            return self.forward_teacher(targets, graph_adjs, decoder_init_hidden, encoder_outputs, masks, meta_value, max_length)
