
import os
import torch
import logging
import argparse

from sven.trainer import PrefixTrainer, TextPromptTrainer
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import MODEL_DIRS

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True) # 输出文件名

    parser.add_argument('--data_dir', type=str, default='../data_train_val') # 训练集路径
    parser.add_argument('--output_dir', type=str, default='../trained/') # 输出路径
    parser.add_argument('--model_type', type=str, default='prefix') # 训练类型
    parser.add_argument('--pretrain_dir', type=str, default=None) # 预训练模型路径
    parser.add_argument('--vul_type', type=str, default=None) 

    parser.add_argument('--n_prefix_token', type=int, default=None) # prefix长度
    parser.add_argument('--num_train_epochs', type=int, default=None) # 训练次数
    parser.add_argument('--kl_loss_ratio', type=int, default=None) # kl速度比率 默认None
    parser.add_argument('--learning_rate', type=float, default=None) # 学习率

    parser.add_argument('--contrastive_loss_ratio', type=int, default=400) # will be divided by 100
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--diff_level', type=str, choices=['prog', 'line', 'char', 'mix'], default='mix')
    parser.add_argument('--lm_loss_ratio', type=int, default=1)

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.pretrain_dir is None: #判断模型位置
        args.pretrain_dir = '2b'
    if args.pretrain_dir in MODEL_DIRS:
        args.pretrain_dir = MODEL_DIRS[args.pretrain_dir]

    if args.n_prefix_token is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.n_prefix_token = 5
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.n_prefix_token = 8
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.n_prefix_token = 12
        else:
            assert False

    if args.num_train_epochs is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.num_train_epochs = 8 #!!!8
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.num_train_epochs = 5
        else:
            assert False

    if args.kl_loss_ratio is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.kl_loss_ratio = 2000
        else:
            assert False

    if args.model_type == 'prefix':
        if args.learning_rate is None:
            args.learning_rate = 1e-2

        if args.contrastive_loss_ratio == 0:
            args.learning_rate = 5e-2
            args.grad_acc_steps = args.grad_acc_steps * 2

        if args.model_type == 'prefix' and args.diff_level in ('prog', 'line'):
            args.learning_rate = 1e-3
    elif args.model_type == 'text':
        args.learning_rate = 5e-5

    args.output_dir = os.path.join(args.output_dir, args.output_name)
    return args

def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_devices(args)
    set_seed(args)

    if args.model_type == 'prefix':
        trainer = PrefixTrainer(args)
    elif args.model_type == 'text':
        trainer = TextPromptTrainer(args)
    else:
        raise NotImplementedError()

    trainer.run()

if __name__ == '__main__':
    main()