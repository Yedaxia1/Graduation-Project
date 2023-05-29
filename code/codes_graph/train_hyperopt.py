# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import copy
import pandas as pd
from collections import defaultdict
from sympy import arg
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.datasets import TUDataset 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric import datasets
from transformer.models import DiffGraphTransformer, GraphTransformer,GNNTransformer
from transformer.data import GraphDataset
from transformer.position_encoding import LapEncoding, FullEncoding, POSENCODINGS
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import pickle
import atexit
from munch import DefaultMunch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_data import FSDataset


def load_args():
    parser = argparse.ArgumentParser(
        description='Transformer baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--nb_heads', type=int, default=4)
    parser.add_argument('--nb_layers', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--pos_enc', choices=[None, 'diffusion', 'pstep', 'adj'], default=None)
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE',default=True)
    parser.add_argument('--lap_dim', type=int, default=8, help='dimension for laplacian PE')
    parser.add_argument('--p', type=int, default=1, help='p step random walk kernel')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='bandwidth for the diffusion kernel')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym',
                        help='normalization for Laplacian')
    parser.add_argument('--dropout', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--layer_norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--zero_diag', action='store_true', help='zero diagonal for PE matrix')
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    return args


def train_epoch(args, model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for i, (data, mask, pe, lap_pe, degree, adjs, labels) in enumerate(loader):
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.lappe:
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(lap_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            lap_pe = lap_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()
            mask = mask.cuda()
            if pe is not None:
                pe = pe.cuda()
            if lap_pe is not None:
                lap_pe = lap_pe.cuda()
            if degree is not None:
                degree = degree.cuda()
            labels = labels.cuda()
            adjs = adjs.cuda()

        optimizer.zero_grad()
        output = model(data, adjs, mask, pe, lap_pe, degree)
        #print(labels)
        labels = labels.reshape(1, -1).squeeze()
        #print(labels)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        pred = output.data.argmax(dim=1)
        running_loss += loss.item() * len(data)
        running_acc += torch.sum(pred == labels).item()

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Train loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_loss


def eval_epoch(args, model, loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    with torch.no_grad():
        for data, mask, pe, lap_pe, degree, adjs, labels in loader:
            if args.lappe:
                # sign flip as in Bresson et al. for laplacian PE
                sign_flip = torch.rand(lap_pe.shape[-1])
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                lap_pe = lap_pe * sign_flip.unsqueeze(0)

            if use_cuda:
                data = data.cuda()
                mask = mask.cuda()
                if pe is not None:
                    pe = pe.cuda()
                if lap_pe is not None:
                    lap_pe = lap_pe.cuda()
                if degree is not None:
                    degree = degree.cuda()
                labels = labels.cuda()
                adjs = adjs.cuda()

            output = model(data, adjs, mask, pe, lap_pe, degree)
            labels = labels.reshape(1, -1).squeeze()
            loss = criterion(output, labels)

            pred = output.data.argmax(dim=1)
            running_loss += loss.item() * len(data)
            running_acc += torch.sum(pred == labels).item()
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Eval loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss

def main(args):
    print(args)
    args = DefaultMunch.fromDict(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # number of node attributes for ZINC dataset
    n_tags = None
    # TUDataset加载方式
    dataset = TUDataset(os.path.join("./TUDataset", args.dataset), name=args.dataset)
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    # dataset = TUDataset(os.path.join("./TUDataset", args.dataset), name=args.dataset)

    # kf = KFold(n_splits=10, shuffle=True, random_state=1)

    all_splits = [k for k in kf.split(dataset, dataset.data.y)]

    train_split, valid_split = all_splits[0]
    train_split, valid_split = train_split.tolist(), valid_split.tolist()
    test_split = valid_split

    train_dset, val_dset, test_dset = dataset[train_split], dataset[valid_split], dataset[test_split]

    num_classes = dataset.num_classes

    # train_dset, val_dset, test_dset, train_split,valid_split,test_split = dataset.kfold_split(test_index=0)
    # dataset = dataset.fc

    train_dset = GraphDataset(train_dset, n_tags, degree=True)
    input_size = train_dset.input_size()
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dset.collate_fn())

    val_dset = GraphDataset(val_dset, n_tags, degree=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dset.collate_fn())

    pos_encoder = None
    if args.pos_enc is not None:
        pos_encoding_method = POSENCODINGS.get(args.pos_enc, None)
        pos_encoding_params_str = ""
        if args.pos_enc == 'diffusion':
            pos_encoding_params = {
                'beta': args.beta
            }
            pos_encoding_params_str = args.beta
        elif args.pos_enc == 'pstep':
            pos_encoding_params = {
                'beta': args.beta,
                'p': args.p
            }
            pos_encoding_params_str = "{}_{}".format(args.p, args.beta)
        else:
            pos_encoding_params = {}

        if pos_encoding_method is not None:
            pos_cache_path = './cache/pe/{}_{}_{}.pkl'.format(args.pos_enc, args.normalization, pos_encoding_params_str)
            pos_encoder = pos_encoding_method(pos_cache_path, normalization=args.normalization, zero_diag=args.zero_diag, **pos_encoding_params)

        print("Position encoding...")
        pos_encoder.apply_to(dataset, split='all')
        train_dset.pe_list = [dataset.pe_list[i] for i in train_split]
        val_dset.pe_list = [dataset.pe_list[i] for i in valid_split]

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
        lap_pos_encoder.apply_to(train_dset)
        lap_pos_encoder.apply_to(val_dset)

    if args.pos_enc is not None:
        model = DiffGraphTransformer(in_size=input_size,
                                     nb_class=2,
                                     d_model=args.dim_hidden,
                                     dim_feedforward=2*args.dim_hidden,
                                     dropout=args.dropout,
                                     nb_heads=args.nb_heads,
                                     nb_layers=args.nb_layers,
                                     batch_norm=args.batch_norm,
                                     lap_pos_enc=args.lappe,
                                     lap_pos_enc_dim=args.lap_dim)
    else:
        model = GNNTransformer(in_size=input_size,
                                 nb_class=2,
                                 d_model=args.dim_hidden,
                                 dim_feedforward=2*args.dim_hidden,
                                 dropout=args.dropout,
                                 nb_heads=args.nb_heads,
                                 nb_layers=args.nb_layers,
                                 lap_pos_enc=args.lappe,
                                 lap_pos_enc_dim=args.lap_dim)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().float()
    criterion.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    test_dset = GraphDataset(test_dset, n_tags, degree=True)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dset.collate_fn())
    if pos_encoder is not None:
        pos_encoder.apply_to(test_dset, split='test')

    if args.lappe and args.lap_dim > 0:
        lap_pos_encoder.apply_to(test_dset)

    print("Training...")
    best_val_acc = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(args,model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
        val_acc, val_loss = eval_epoch(args,model, val_loader, criterion, args.use_cuda)
        test_acc, test_loss = eval_epoch(args,model, test_loader, criterion, args.use_cuda)

        if args.warmup is None:
            lr_scheduler.step(val_loss)

        logs['train_loss'].append(train_loss)
        logs['val_acc'].append(val_acc)
        logs['val_loss'].append(val_loss)
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    total_time = timer() - start_time
    print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_acc, test_loss = eval_epoch(args, model, test_loader, criterion, args.use_cuda)

    print("test Acc {:.4f}".format(test_acc))
    return {
        "loss": -best_val_acc,
        'status': STATUS_OK,
        'params': args
    }


if __name__ == "__main__":
    def save_result(result_file,trials):
        print("正在保存结果...")
        with open(result_file, "w+") as f:
            for result in trials.results:
                if 'loss' in result and result['loss'] <= trials.best_trial['result']['loss']:
                    print(result, file=f)
        print("结果已保存 {:s}".format(result_file))
        print(trials.best_trial)
    def initial_hyperopt(trial_file,result_file,max_evals):
        try:
            with open(trial_file, "rb") as f:
                trials = pickle.load(f)
            current_process = len(trials.results)
            print("使用已有的trial记录, 现有进度: {:d}/{:d} {:s}".format(current_process,max_evals,trial_file))
        except:
            trials = Trials()
            print("未找到现有进度, 从0开始训练 0/{:d} {:s}".format(max_evals, trial_file))
        atexit.register(save_result,result_file,trials)
        return trials

    max_evals = 200
    args = vars(load_args())
    # args['pos_enc'] = hp.choice('pos_enc',['diffusion','pstep','adj'])
    args['nb_heads'] = hp.choice('nb_heads',[4,8])
    args['nb_layers'] = hp.choice('nb_layers',[3,5,7])
    args['lr'] = hp.choice('lr',[0.01, 0.005, 0.001])
    args['weight_decay'] = hp.choice('weight_decay',[0, 1e-4, 5e-4, 1e-5])
    args['dropout'] = hp.choice('dropout',[0, 0.3, 0.5, 0.8])
    

    save_root = os.path.join("hyperopt")
    result_file = os.path.join(save_root, f"result.log")
    trial_file = os.path.join(save_root, f"result.trial")

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    trials = initial_hyperopt(trial_file,result_file,max_evals)
    best = fmin(
        fn=main,space=args, algo=tpe.suggest, max_evals=max_evals, 
        trials = trials, trials_save_file=trial_file)
