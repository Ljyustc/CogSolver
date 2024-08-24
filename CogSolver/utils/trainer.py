# -*- coding: utf-8 -*-

import random
import logging
import torch
from torch import nn, optim

from utils import Checkpoint, Evaluator

class SupervisedTrainer(object):
    def __init__(self, class_dict, class_list, use_cuda):
        self.test_train_every = 10
        self.print_every = 30
        self.use_cuda = use_cuda

        self.pad_idx_in_class = class_dict['PAD_token']

        loss_weight = torch.ones(len(class_dict))
        loss_weight[self.pad_idx_in_class] = 0
        self.loss = nn.NLLLoss(weight=loss_weight, reduction="sum")
        if use_cuda:
            self.loss = self.loss.cuda()
        
        self.evaluator = Evaluator(
            class_dict=class_dict,
            class_list=class_list,
            use_cuda=use_cuda
        )
        return

    def _train_batch(self, input_variables, num_pos, input_lengths, span_length, target_variables, tree, model, batch_size, regular=0):
        decoder_outputs, _, _, node_hidden_pairs = model(
            input_variable=input_variables,
            num_pos=num_pos,
            input_lengths=input_lengths, 
            span_length=span_length,
            target_variable=target_variables, 
            tree=tree
        )

        batch_size = span_length.size(0)

        # loss
        loss = 0
        for step, step_output in enumerate(decoder_outputs):
            loss += self.loss(step_output.contiguous().view(batch_size, -1), target_variables[:, step].view(-1))
        
        total_target_length = (target_variables != self.pad_idx_in_class).sum().item()
        loss = loss / total_target_length
        loss += regular * sum(node_hidden_pairs) / len(node_hidden_pairs)
        
        assert torch.isnan(loss).sum() == 0
        
        model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_epoches(self, data_loader, model, batch_size, start_epoch, start_step, max_acc, regular, n_epoch):
        train_list = data_loader.train_list
        test_list = data_loader.test_list

        step = start_step
        print_loss_total = 0
        max_ans_acc = max_acc

        for epoch_index, epoch in enumerate(range(start_epoch, n_epoch + 1)):
            model.train()
            # random.shuffle(train_list)
            batch_generator = data_loader.get_batch(train_list, batch_size, template_flag=True)
            for batch_data_dict in batch_generator:
                step += 1
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

                target_variables = batch_data_dict['batch_decode_idx']
                target_variables = torch.LongTensor(target_variables)
                if self.use_cuda:
                    target_variables = target_variables.cuda()

                loss = self._train_batch(
                    input_variables=input_variables,
                    num_pos=num_pos, 
                    input_lengths=input_lengths,
                    span_length=span_length, 
                    target_variables=target_variables,
                    tree=tree,                    
                    model=model,
                    batch_size=batch_size,
                    regular=regular
                )

                print_loss_total += loss
                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    logging.info(f'step: {step}, Train loss: {print_loss_avg:.4f}')
                    if self.use_cuda:
                        torch.cuda.empty_cache()
            self.scheduler.step()

            model.eval()
            with torch.no_grad():
                test_temp_acc, test_ans_acc = self.evaluator.evaluate(
                    model=model,
                    data_loader=data_loader,
                    data_list=test_list,
                    template_flag=True,
                    template_len=True,
                    batch_size=batch_size,
                )
                if epoch_index % self.test_train_every == 0:
                    train_temp_acc, train_ans_acc = self.evaluator.evaluate(
                        model=model,
                        data_loader=data_loader,
                        data_list=train_list,
                        template_flag=True,
                        template_len=True,
                        batch_size=batch_size,
                    )

                    logging.info(f"Epoch: {epoch}, Step: {step}, test_acc: {test_temp_acc:.3f}, {test_ans_acc:.3f}, train_acc: {train_temp_acc:.3f}, {train_ans_acc:.3f}")
                else:
                    logging.info(f"Epoch: {epoch}, Step: {step}, test_acc: {test_temp_acc:.3f}, {test_ans_acc:.3f}")
            
            if test_ans_acc > max_ans_acc:
                max_ans_acc = test_ans_acc
                logging.info("saving checkpoint ...")
                Checkpoint.save(epoch=epoch, step=step, max_acc=max_ans_acc, model=model, optimizer=self.optimizer, scheduler=self.scheduler, best=True)
            else:
                Checkpoint.save(epoch=epoch, step=step, max_acc=max_ans_acc, model=model, optimizer=self.optimizer, scheduler=self.scheduler, best=False)
        return

    def train(self, model, data_loader, batch_size, regular, n_epoch, resume=False, 
              optim_lr=1e-3, optim_weight_decay=1e-5, scheduler_step_size=60, scheduler_gamma=0.6):
        start_epoch = 1
        start_step = 0
        max_acc = 0
        self.optimizer = optim.Adam(model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        if resume:
            resume_checkpoint = Checkpoint.load(model_only=False)
            model.load_state_dict(resume_checkpoint.model)
            resume_optimizer = resume_checkpoint.optimizer
            resume_scheduler = resume_checkpoint.scheduler
            if resume_optimizer is not None:
                start_epoch = resume_checkpoint.epoch
                start_step = resume_checkpoint.step
                max_acc = resume_checkpoint.max_acc
                self.optimizer.load_state_dict(resume_optimizer)
                self.scheduler.load_state_dict(resume_scheduler)

        self._train_epoches(
            data_loader=data_loader, 
            model=model, 
            batch_size=batch_size,
            start_epoch=start_epoch, 
            start_step=start_step, 
            max_acc=max_acc,
            regular=regular,
            n_epoch=n_epoch
        )
        return
