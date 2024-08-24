# -*- coding: utf-8 -*-

import json
import logging
import torch

from .equ_tools import pre_solver

class Evaluator(object):
    def __init__(self, class_dict, class_list, use_cuda=False):
        self.class_list = class_list
        self.use_cuda = use_cuda

        self.pad_idx_in_class = class_dict['PAD_token']
        self.end_idx_in_class = class_dict['END_token']
        return

    def inverse_temp_to_num(self, equ_list, num_list):
        equ_list = [num_list[ord(token[-1]) - ord('a')] if "temp_" in token else 3.14 if token == 'PI' else token for token in equ_list]
        return equ_list

    def decode_token(self, seq_var):
        decode_list = []
        for idx in seq_var.cpu().tolist():
            if idx != self.pad_idx_in_class and idx != self.end_idx_in_class:
                decode_list.append(self.class_list[idx])
            else:
                break
        return decode_list

    def compute_gen_ans(self, seq_var, num_list):
        decode_list = self.decode_token(seq_var)
        try:
            # exception: predicted temp_* not appear in text
            equ_list = self.inverse_temp_to_num(decode_list, num_list)
            if "END_token" in equ_list:
                equ_list = equ_list[:equ_list.index('END_token')]
            # exception: invalid prefix expression
            ans = pre_solver(equ_list)
            if ans is None:
                ans = "error"
        except:
            ans = "error"
        return decode_list, ans

    def evaluate(self, model, data_loader, data_list, template_flag, template_len, batch_size, beam_width=None, test_log=None, print_probability=True):
        temp_count = 0
        acc_right = 0
        total_num = len(data_list)

        eval_report = {}

        batch_generator = data_loader.get_batch(data_list, batch_size, template_flag or template_len)
        for batch_data_dict in batch_generator:
            input_variables = batch_data_dict['batch_span_encode_idx']
            input_lengths = batch_data_dict['batch_span_encode_len']
            span_length = batch_data_dict['batch_span_len']
            tree = batch_data_dict["batch_tree"]
            
            input_variables = [torch.LongTensor(input_variable) for input_variable in input_variables]
            input_lengths = [torch.LongTensor(input_length) for input_length in input_lengths]
            span_length = torch.LongTensor(span_length)
            if self.use_cuda:
                input_variables = [input_variable.cuda() for input_variable in input_variables]
                input_lengths = [input_length.cuda() for input_length in input_lengths]
                span_length = span_length.cuda()
        
            span_num_pos = batch_data_dict["batch_span_num_pos"]
            word_num_poses = batch_data_dict["batch_word_num_poses"]
            span_num_pos = torch.LongTensor(span_num_pos)
            word_num_poses = [torch.LongTensor(word_num_pos) for word_num_pos in word_num_poses]
            if self.use_cuda:
                span_num_pos = span_num_pos.cuda()
                word_num_poses = [word_num_pose.cuda() for word_num_pose in word_num_poses]
            num_pos = (span_num_pos, word_num_poses)
            
            batch_index = batch_data_dict['batch_index']
            batch_text = batch_data_dict['batch_text']
            batch_num_list = batch_data_dict['batch_num_list']
            batch_solution = batch_data_dict['batch_solution']
            batch_size = len(batch_index)

            if template_flag or template_len:
                target_variables = batch_data_dict['batch_decode_idx']
                target_variables = torch.LongTensor(target_variables)
                if template_flag:
                    if self.use_cuda:
                        target_variables = target_variables.cuda()
                else:
                    target_variables = None
                if template_len:
                    max_length = target_variables.size(1)
                else:
                    max_length = None
            else:
                target_variables = None
                max_length = None

            decoder_outputs, _, symbols_list = model(
                input_variable=input_variables,
                num_pos=num_pos,
                input_lengths=input_lengths,
                span_length=span_length,
                target_variable=None,
                tree=tree,
                max_length=max_length,
                beam_width=beam_width
            )

            # batch_size * seq_len
            seq_var = torch.cat(symbols_list, dim=1)
            decoder_outputs_prob = [torch.exp(decoder_output) for decoder_output in decoder_outputs]
            # answer acc
            for i in range(batch_size):
                decode_list, gen_ans = self.compute_gen_ans(seq_var[i], batch_num_list[i])
                target_ans = batch_solution[i]
                if gen_ans == 'error':
                    ans_status = "invalid"
                else:
                    try:
                        if abs(float(gen_ans) - float(target_ans)) <1e-5:
                            acc_right += 1
                            ans_status = "correct"
                        else:
                            ans_status = "error"
                    except:
                        logging.info(f"gen_ans: {gen_ans}, target_ans: {target_ans}")
                
                # error log dump
                if test_log is not None:
                    predict_entry = {}
                    if template_flag:
                        target_list = self.decode_token(target_variables[i])
                    
                    predict_entry["text"] = batch_text[i]
                    predict_entry["expr"] = ' '.join(decode_list)
                    predict_entry["target"] = ' '.join(target_list)
                    predict_entry["predict_ans"] = str(gen_ans)
                    predict_entry["true_ans"] = str(target_ans)
                    if print_probability:
                        if len(decode_list) > 0:
                            prob = torch.cat([decoder_outputs_prob[si][i].unsqueeze(0) for si in range(len(decode_list))], dim=0).cpu().tolist()
                        else:
                            prob = []
                    predict_entry["prob"] = json.dumps(prob)
                    predict_entry["ans_status"] = ans_status
                    eval_report[batch_index[i]] = predict_entry
            
            # template acc
            if template_flag:
                seq_var_len = seq_var.size(1)
                target_var_len = target_variables.size(1)
                for i in range(batch_size):
                    right_flag = 0
                    for j in range(target_var_len):
                        if j == seq_var_len:
                            if target_variables[i][j].item() == self.end_idx_in_class:
                                right_flag = 1
                            break
                        elif seq_var[i][j].item() == self.end_idx_in_class and target_variables[i][j].item() == self.end_idx_in_class:
                            right_flag = 1
                            break
                        elif target_variables[i][j].item() != seq_var[i][j].item():
                            break
                    temp_count += right_flag
                    if test_log is not None:
                        eval_report[batch_index[i]]["expr_status"] = "correct" if right_flag == 1 else "error"
            if self.use_cuda:
                torch.cuda.empty_cache()
    
        ans_acc = acc_right / total_num
        if template_flag:
            temp_acc = temp_count / total_num
        else:
            temp_acc = 0

        if test_log is not None:
            with open(test_log, "wt", encoding="utf-8") as file:
                json.dump(eval_report, file, ensure_ascii=False, indent=4)
        return temp_acc, ans_acc
