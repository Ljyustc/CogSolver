# -*- coding: utf-8 -*-

import os
import logging
import random
import torch
import torch.nn as nn

from config import get_args
from model import Encoder, Knowledge_Base, Decoder, Seq2seq
from utils import DataLoader, Checkpoint, Evaluator, SupervisedTrainer

def init():
    args = get_args()
    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
        if not torch.cuda.is_available():
            args.use_cuda = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename=args.log)
    logging.info('\n' + '\n'.join([f"\t{'['+k+']':20}\t{v}" for k, v in dict(args._get_kwargs()).items()]))

    checkpoint_path = os.path.join("./experiment", args.checkpoint)
    if not os.path.exists(checkpoint_path):
        logging.info(f'create checkpoint directory {checkpoint_path} ...')
        os.makedirs(checkpoint_path)
    Checkpoint.set_ckpt_path(checkpoint_path)
    return args

def create_model(args):
    trim_min_count = 5
    data_loader = DataLoader(args, trim_min_count=trim_min_count)

    embed_model = nn.Embedding(data_loader.vocab_len, args.embed)
    embed_model.weight.data.copy_(data_loader.embed_vectors)
    encode_model = Encoder(
        embed_model=embed_model,
        hidden_size=args.hidden,
        word_know_threshold=args.word_know_threshold,
        span_size=data_loader.span_size,
        dropout=args.dropout,
    )
    
    knowledge_base = Knowledge_Base(
        embed_model=embed_model,
        op_set=data_loader.op_set, 
        vocab_dict=data_loader.vocab_dict,
        op_threshold=args.op_threshold,
        use_cuda=args.use_cuda
    )

    decode_model = Decoder(
        embed_model=embed_model,
        op_set=data_loader.op_set,
        vocab_dict=data_loader.vocab_dict,
        class_list=data_loader.class_list,
        ft=args.ft,
        hidden_size=args.hidden,
        dropout=args.dropout,
        use_cuda=args.use_cuda,
        knowledge_base=knowledge_base
    )
    
    seq2seq = Seq2seq(encode_model, knowledge_base, decode_model)
    return seq2seq, data_loader

def train(args):
    seq2seq, data_loader = create_model(args)
    if args.use_cuda:
        seq2seq = seq2seq.cuda()

    st = SupervisedTrainer(
        class_dict=data_loader.class_dict,
        class_list=data_loader.class_list,
        use_cuda=args.use_cuda
    )

    logging.info('start training ...')
    st.train(
        model=seq2seq, 
        data_loader=data_loader,
        batch_size=args.batch,
        regular=args.regular,
        n_epoch=args.epoch,
        resume=args.resume,
        optim_lr=args.lr,
        optim_weight_decay=args.weight_decay,
        scheduler_step_size=args.step,
        scheduler_gamma=args.gamma,
    )
    return

def test(args, test_dataset="test"):
    seq2seq, data_loader = create_model(args)
    if args.use_cuda:
        seq2seq = seq2seq.cuda()
    resume_checkpoint = Checkpoint.load(model_only=True)
    seq2seq.load_state_dict(resume_checkpoint.model)

    evaluator = Evaluator(
        class_dict=data_loader.class_dict,
        class_list=data_loader.class_list,
        use_cuda=args.use_cuda
    )
    if test_dataset == "test":
        test_dataset = data_loader.test_list
    elif test_dataset == "train":
        test_dataset = data_loader.train_list
    seq2seq.eval()
    with torch.no_grad():
        test_temp_acc, test_ans_acc = evaluator.evaluate(
            model=seq2seq,
            data_loader=data_loader,
            data_list=test_dataset,
            template_flag=True,
            template_len=False,
            batch_size=args.batch,
            beam_width=args.beam,
            test_log=args.test_log,
            print_probability=True
        )
    logging.info(f"temp_acc: {test_temp_acc}, ans_acc: {test_ans_acc}")
    return

if __name__ == "__main__":
    args = init()
    if args.run_flag == "test":
        test(args, "test")
    elif args.run_flag == 'train':
        train(args)
    else:
        logging.info('unknown run_flag')
