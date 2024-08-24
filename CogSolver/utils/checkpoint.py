# -*- coding:utf-8 -*-

import logging
import os
import torch

class Checkpoint():

    CHECKPOINT_NAME = 'checkpoint'
    TRAINER_NAME = 'trainer.ckpt'
    MODEL_NAME = 'model.ckpt'

    def __init__(self, epoch, step, max_acc, model, optimizer, scheduler):
        self.epoch = epoch
        self.step = step
        self.max_acc = max_acc
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        return

    @classmethod
    def set_ckpt_path(cls, path):
        cls.CHECKPOINT_NAME = path
        return

    @classmethod
    def save(cls, epoch=1, step=0, max_acc=0, model=None, optimizer=None, scheduler=None, best=False):
        ckpt_path = cls.CHECKPOINT_NAME
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        if model is not None:
            saved_model = model.state_dict()
        else:
            saved_model = None
        if optimizer is not None:
            saved_optimizer = optimizer.state_dict()
        else:
            saved_optimizer = None
        if scheduler is not None:
            saved_scheduler = scheduler.state_dict()
        else:
            saved_scheduler = None
        
        if best:
            torch.save(saved_model, os.path.join(ckpt_path, cls.MODEL_NAME))
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "max_acc": max_acc,
                "model": saved_model,
                "optimizer": saved_optimizer,
                "scheduler": saved_scheduler,
            },
            os.path.join(ckpt_path, cls.TRAINER_NAME)
        )
        return

    @classmethod
    def load(cls, model_only=False):
        ckpt_path = cls.CHECKPOINT_NAME
        logging.info(f"Loading checkpoints from {ckpt_path} ...")
        model_path = os.path.join(ckpt_path, cls.MODEL_NAME)
        trainer_path = os.path.join(ckpt_path, cls.TRAINER_NAME)
        if model_only or (not os.path.exists(trainer_path)):
            resume_epoch = 1
            resume_step = 0
            resume_max_acc = 0
            resume_model = torch.load(model_path)
            resume_optimizer = None
            resume_scheduler = None
        else:
            resume_trainer = torch.load(trainer_path)
            resume_epoch = resume_trainer["epoch"]
            resume_step = resume_trainer["step"]
            resume_max_acc = resume_trainer["max_acc"]
            resume_model = resume_trainer["model"]
            resume_optimizer = resume_trainer["optimizer"]
            resume_scheduler = resume_trainer["scheduler"]
        checkpoint = Checkpoint(
            epoch=resume_epoch,
            step=resume_step,
            max_acc=resume_max_acc,
            model=resume_model,
            optimizer=resume_optimizer,
            scheduler=resume_scheduler
        )
        return checkpoint
