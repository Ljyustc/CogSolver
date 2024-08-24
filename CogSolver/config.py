# -*- coding:utf-8 -*-

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Hierarchical Math Solver')

    parser.add_argument('--cuda', type=str, dest='cuda_id', default=None)
    parser.add_argument('--checkpoint', type=str, dest='checkpoint', default=None)
    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    parser.add_argument('--log', type=str, dest='log', default=None)
    parser.add_argument('--test-log', type=str, dest='test_log', default=None)
    parser.add_argument('--seed', type=int, dest='seed', default=10)
    parser.add_argument('--run-flag', type=str, dest='run_flag',default='train')

    parser.add_argument('--learn_iteration', type=int, dest='epoch', default=120)
    parser.add_argument('--batch', type=int, dest='batch', default=64)
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3)
    parser.add_argument('--weight-decay', type=float, dest='weight_decay', default=1e-5)
    parser.add_argument('--step', type=int, dest='step', default=20)
    parser.add_argument('--gamma', type=float, dest='gamma', default=0.5)
    parser.add_argument('--beam', type=int, dest='beam', default=1)
    parser.add_argument('--word_know_threshold', type=float, dest='word_know_threshold', default=0.7)
    parser.add_argument('--op_threshold', type=float, dest='op_threshold', default=0.3)
    parser.add_argument('--lambda_for_commutative_loss', type=float, dest='regular', default=1e-3)
    parser.add_argument('--forget_gate_threshold', type=float, dest='ft', default=0.99)

    parser.add_argument('--embed', type=int, dest='embed', default=128)
    parser.add_argument('--hidden', type=int, dest='hidden', default=512)
    parser.add_argument('--dropout', type=float, dest='dropout', default=0.5)

    args = parser.parse_args()

    args.use_cuda = args.cuda_id is not None
    return args
