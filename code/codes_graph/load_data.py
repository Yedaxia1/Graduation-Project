import pickle
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from scipy import io
from torch.utils.data import Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from glob import glob
from code_utils import *
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices


class FSDataset(object):
    def __init__(self, root_dir, folds, seed, thresh_persent=None, n_neighbors=None):
        '''
        # MDD数据集加载
        data_files = glob(os.path.join(root_dir,"**","*.pkl"))
        data_files.sort()
        self.fc = []

        for file in data_files:
            with open(file, "rb") as f:
                data =  pickle.load(f)
            adj = data.adjacency_mat
            feature = data.source_mat.T

            feature = np.corrcoef(feature)

            # 取ROISignals_S(1)-1-0001的括号部分，1是MDD，2是NC，减一变成二分类用的0,1标签
            label = int(re.match(r".*ROISignals_.+-(\d)-.+?\.pkl",file).group(1)) - 1
            fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
            self.fc.append(Data(
                x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
                ))

        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)
        '''
        
        '''
        # HCP数据集加载
        data_files = glob(os.path.join(root_dir,"**", "**", "*.pkl"))
        data_files.sort()
        self.fc = []

        for file in data_files:
            f = open(file, "rb")
            # print(file)
            data =  pickle.load(f)

            f.close()
            adj = data.adjacency_mat
            feature = data.source_mat.T

            feature = np.corrcoef(data.source_mat.T)

            label = 0 if file.split('\\')[-3]=="female" else 1
            print(file, label)
            # 取ROISignals_S(1)-1-0001的括号部分，1是MDD，2是NC，减一变成二分类用的0,1标签
            
            fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
            self.fc.append(Data(
                x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
                ))
        
        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)
        '''

        
        # zhongda,xinxiang数据集加载
        data_files = glob(os.path.join(root_dir,"**", "**", "**", "*.pkl"))
        data_files.sort()
        self.fc = []

        remove_file = [
            'zhongda_data_fmri/HC/HC_043',
            'zhongda_data_fmri/UMD/NRD_057'
        ]

        for file in data_files:
            case_name = re.match(r".*zhongda_xinxiang_data.(.*).calStaticRes.pkl",file).group(1)
            if case_name in remove_file:
                continue

            with open(file, "rb") as f:
                data =  pickle.load(f)
            
            feature = data.source_mat.T

            feature = np.corrcoef(data.source_mat.T)

            # if thresh_persent is None:
            #     adj = data.adjacency_mat
            # else:
            #     adj = convert_binary_by_thresh_val(feature, get_thresh_val(feature, thresh_persent))
            if n_neighbors is None:
                adj = knn_generate_graph(data.source_mat)
            else:
                adj = knn_generate_graph(data.source_mat, n_neighbors=n_neighbors)

            # 取ROISignals_S(1)-1-0001的括号部分，1是MDD，2是NC，减一变成二分类用的0,1标签
            label = 1
            if 'HC' in file:
                label = 0
            # print(file, label)
            fcedge_index, _ = dense_to_sparse(torch.from_numpy(adj.astype(np.int16)))
            self.fc.append(Data(
                x=torch.from_numpy(feature).float(), edge_index=fcedge_index, y=torch.as_tensor(label).long()
                ))

        self.k_fold = folds
        self.k_fold_split = K_Fold(self.k_fold, self.fc, seed)
        
        

    def kfold_split(self, test_index):
        assert test_index < self.k_fold
        # valid_index = (test_index + 1) % self.k_fold
        valid_index = test_index 
        test_split = self.k_fold_split[test_index]
        valid_split = self.k_fold_split[valid_index]

        train_mask = np.ones(len(self.fc))
        train_mask[test_split] = 0
        train_mask[valid_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.fc, train_split.tolist())
        valid_subset = Subset(self.fc, valid_split.tolist())
        test_subset = Subset(self.fc, test_split.tolist())

        return train_subset, valid_subset, test_subset, train_split,valid_split,test_split
    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.fc)

if __name__ == "__main__":
#   data = io.loadmat(r"..\data_graph\MDD\ROISignals_FunImgARCWF\ROISignals_S1-1-0001\ROISignals_S1-1-0001.mat")
#   data = io.loadmat(r"..\data_graph\MDD\ROISignals_FunImgARCWF\ROISignals_S1-1-0005\ROISignals_S1-1-0005.mat")
  data = io.loadmat(r"D:\Project\python\graphmix\data_graph\MDD\data_90_230\len_90\ROISignals_S5-1-0001.mat")
  print(len(data['ROISignals'][0]),len(data['ROISignals']))