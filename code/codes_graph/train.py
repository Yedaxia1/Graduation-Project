# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import copy
import pandas as pd
from collections import defaultdict
import torch
from transformer.utils import sensitivity_specificity
from code_utils import * 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold, StratifiedKFold
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric import datasets 
from transformer.models import DiffGraphTransformer, GraphTransformer,GNNTransformer
from transformer.data import GraphDataset
from transformer.position_encoding import LapEncoding, FullEncoding, POSENCODINGS
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim
from torch_geometric.datasets import TUDataset 
from baseline.GAT import GAT
from baseline.GCN import GCN
from baseline.GIN import GIN
from baseline.GraphSAGE import GraphSAGE
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib

from utils import NormalizedDegree, k_fold
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
    parser.add_argument('--lappe', action='store_true', help='use laplacian PE',default=False)
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
    # parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, num_classes, use_cuda=False):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for i, (data, mask, pe, lap_pe, degree, adjs, labels) in enumerate(loader):
        labels = labels.squeeze_()
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
        labels = labels.reshape(1, -1).squeeze()
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


def eval_epoch(model, loader, criterion, use_cuda=False, num_classes=2):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    sen, sp, f1, auc = 0, 0, 0.0, 0.0

    tic = timer()
    with torch.no_grad():
        for data, mask, pe, lap_pe, degree, adjs, labels in loader:
            labels = labels.squeeze_()
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
            sensitivity, specificity = sensitivity_specificity(labels.cpu(),pred.cpu(), num_classes)
            if num_classes==2:
                val_f1_score = f1_score(labels.cpu(),pred.cpu())
                val_auc = roc_auc_score(labels.cpu(),pred.cpu())

            sen += sensitivity * len(data)
            sp += specificity * len(data)
            if num_classes==2:
                f1 += val_f1_score * len(data)
                auc += val_auc * len(data)

            running_loss += loss.item() * len(data)
            running_acc += torch.sum(pred == labels).item()
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample

    print('Eval loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss, sen/ n_sample, sp/ n_sample, f1/ n_sample, auc/ n_sample

def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    n_tags = None
    random_s = np.array([25, 50, 100, 125, 150, 175, 200, 225, 250, 275], dtype=int)

    thresh_persent_array = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3], dtype=float)

    layer_num_array = np.array([1,2,3,4,5,6,7,8,9,10], dtype=int)

    n_neighbors_array = np.array([6,7,8,9,11,12,13,14,16,17,18,19], dtype=int)

    
    test_acc_list, test_sp_list,test_sen_list = [],[],[]
    best_acc_list, best_sp_list,best_sen_list = [],[],[]
    # 'NCI1', 'IMDB-BINARY', 'PTC_FR', 'MUTAG', 'IMDB-MULTI', 'NCI109', 
    dataset_names_array = np.array(['IMDB-MULTI'], dtype=str)
    
    for dataset_name in dataset_names_array:

        for layer_num in layer_num_array:
            best_acc_list_pertime, best_sp_list_pertime,best_sen_list_pertime,  best_f1_list_pertime,best_auc_list_pertime= [],[],[],[],[]

            # dataset = FSDataset('/media/zhangke/data_disk/Topology_data/zhongda_xinxiang_data',folds=10,seed=random_s[cross_valid_time])
            # 'NCI1', 'IMDB-BINARY', 'PTC_FR', 'MUTAG', 'IMDB-MULTI'
            dataset = TUDataset(os.path.join("./TUDataset", dataset_name), name=dataset_name)
            dataset.data.edge_attr = None
            if dataset.data.x is None:
                max_degree = 0
                degs = []
                for data in dataset:
                    degs += [degree(data.edge_index[0], dtype=torch.long)]
                    max_degree = max(max_degree, degs[-1].max().item())

                if max_degree < 1000:
                    dataset.transform = T.OneHotDegree(max_degree)
                else:
                    deg = torch.cat(degs, dim=0).to(torch.float)
                    mean, std = deg.mean().item(), deg.std().item()
                    dataset.transform = NormalizedDegree(mean, std)

            train_indices, test_indices, val_indices = k_fold(dataset, 10)

            print("dataset {} start.".format(dataset_name))

            for i in range(10):
                train_split, test_split, valid_split = train_indices[i], test_indices[i], val_indices[i]

                train_dset, val_dset, test_dset = dataset[train_split], dataset[valid_split], dataset[test_split]

                num_classes = dataset.num_classes
                

                # train_dset, val_dset, test_dset, train_split,valid_split,test_split = dataset.kfold_split(test_index=i)
                
                # dataset = dataset.fc
                # num_classes = 2

                train_dset = GraphDataset(train_dset, n_tags, degree=True)
                input_size = train_dset.input_size()
                train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dset.collate_fn())

                val_dset = GraphDataset(val_dset, n_tags, degree=True)
                val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, collate_fn=val_dset.collate_fn())
                
                test_dset = GraphDataset(test_dset, n_tags, degree=True)
                test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=True, collate_fn=test_dset.collate_fn())

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
                        pos_cache_path = './cache/pe/{}/{}_{}_{}.pkl'.format(dataset_name, args.pos_enc, args.normalization, pos_encoding_params_str)
                        pos_encoder = pos_encoding_method(pos_cache_path, normalization=args.normalization, zero_diag=args.zero_diag, **pos_encoding_params)

                    print("Position encoding...")
                    pos_encoder.apply_to(dataset, split='all')
                    train_dset.pe_list = [dataset.pe_list[i] for i in train_split]
                    val_dset.pe_list = [dataset.pe_list[i] for i in valid_split]
                    test_dset.pe_list = [dataset.pe_list[i] for i in test_split]
                    

                if args.lappe and args.lap_dim > 0:
                    lap_pos_encoder = LapEncoding(args.lap_dim, normalization='sym')
                    lap_pos_encoder.apply_to(train_dset)
                    lap_pos_encoder.apply_to(val_dset)

                if args.pos_enc is not None:
                    # model = DiffGraphTransformer(in_size=input_size,
                    #                             nb_class=2,
                    #                             d_model=args.dim_hidden,
                    #                             dim_feedforward=2*args.dim_hidden,
                    #                             dropout=args.dropout,
                    #                             nb_heads=args.nb_heads,
                    #                             nb_layers=args.nb_layers,
                    #                             batch_norm=args.batch_norm,
                    #                             lap_pos_enc=args.lappe,
                    #                             lap_pos_enc_dim=args.lap_dim)
                    model = GNNTransformer(in_size=input_size,
                                            nb_class=num_classes,
                                            d_model=args.dim_hidden,
                                            dim_feedforward=2*args.dim_hidden,
                                            dropout=args.dropout,
                                            nb_heads=args.nb_heads,
                                            nb_layers=layer_num,
                                            lap_pos_enc=args.lappe,
                                            lap_pos_enc_dim=args.lap_dim)
                    
                else:
                    # model = GraphTransformer(in_size=input_size,
                    #                         nb_class=2,
                    #                         d_model=args.dim_hidden,
                    #                         dim_feedforward=2*args.dim_hidden,
                    #                         dropout=args.dropout,
                    #                         nb_heads=args.nb_heads,
                    #                         nb_layers=args.nb_layers,
                    #                         lap_pos_enc=args.lappe,
                    #                         lap_pos_enc_dim=args.lap_dim)
                    model = GNNTransformer(in_size=input_size,
                                            nb_class=num_classes,
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
                class_weight = compute_class_weight('balanced', np.unique(dataset.data.y.numpy()),dataset.data.y.numpy())
                criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight)).float()
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


                # if pos_encoder is not None:
                #     pos_encoder.apply_to(test_dset, split='test')
                #     test_dset.pe_list = [dataset.pe_list[i] for i in test_split]

                if args.lappe and args.lap_dim > 0:
                    lap_pos_encoder.apply_to(test_dset)

                print("Training...")
                best_val_acc = 0
                # best_model = None
                best_epoch = 0
                start_time = timer()
                for epoch in range(args.epochs):
                    print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
                    train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, num_classes, args.use_cuda)
                    val_acc, val_loss, val_sen, val_sp, f1_s, auc_s = eval_epoch(model, val_loader, criterion, args.use_cuda, num_classes)
                    # test_acc, test_loss, sensitivity, specificity = eval_epoch(model, test_loader, criterion, args.use_cuda)

                    if args.warmup is None:
                        lr_scheduler.step(val_loss)

                    if val_acc > best_val_acc:
                        best_val_acc, best_sp, best_sen, best_f1, best_auc =  val_acc, val_sp, val_sen, f1_s, auc_s
                        best_epoch = epoch
                        best_weights = copy.deepcopy(model.state_dict())

                total_time = timer() - start_time
                print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))
                model.load_state_dict(best_weights)

                print("Testing...")
                test_acc, test_loss, sensitivity, specificity, f1_s, auc_s = eval_epoch(model, test_loader, criterion, args.use_cuda, num_classes)

                print("test Acc {:.4f}".format(test_acc))

                best_acc_list_pertime.append(test_acc)
                best_sp_list_pertime.append(specificity)
                best_sen_list_pertime.append(sensitivity)
                best_f1_list_pertime.append(f1_s)
                best_auc_list_pertime.append(auc_s)
            
            
            print("{} 10 fold test average: {:.2f}±{:.2f} sensitivity:{:.2f}±{:.2f} specificity:{:.2f}±{:.2f} f1_score:{:.2f}±{:.2f} auc:{:.2f}±{:.2f}".format(
                dataset_name,
                np.mean(best_acc_list_pertime)*100, np.std(best_acc_list_pertime)*100,
                np.mean(best_sen_list_pertime)*100, np.std(best_sen_list_pertime)*100,
                np.mean(best_sp_list_pertime)*100, np.std(best_sp_list_pertime)*100,
                np.mean(best_f1_list_pertime)*100, np.std(best_f1_list_pertime)*100,
                np.mean(best_auc_list_pertime)*100, np.std(best_auc_list_pertime)*100,
            ))

            with open('./results/{}_lay_res.txt'.format(dataset_name), "a+", encoding='utf-8') as rf:
                rf.write(
                    'dataset: %s layer_num: %s 10 fold test average: %.2f±%.2f, sensitivity: %.2f±%.2f, specificity: %.2f±%.2f, f1: %.2f±%.2f, auc: %.2f±%.2f \n' % (
                        dataset_name,
                        layer_num,
                        np.mean(best_acc_list_pertime)*100, np.std(best_acc_list_pertime)*100,
                        np.mean(best_sen_list_pertime)*100, np.std(best_sen_list_pertime)*100,
                        np.mean(best_sp_list_pertime)*100, np.std(best_sp_list_pertime)*100,
                        np.mean(best_f1_list_pertime)*100, np.std(best_f1_list_pertime)*100,
                        np.mean(best_auc_list_pertime)*100, np.std(best_auc_list_pertime)*100,
                    )
                )
                






if __name__ == "__main__":
    main()
