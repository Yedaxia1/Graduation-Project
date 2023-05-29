import pickle as pkl
from glob import glob
import numpy as np
import os
import numpy as np

from code_utils import knn_generate_graph
src = np.random.randint(0, 100, (10,5))
import pickle 

# feature = np.corrcoef(src.T)
# print(src.shape[1])
# A = knn_generate_graph(src, 2)
# print(A)
# a = 0.2
# best_acc_list = [1,2,2,3]
# best_sen_list= [1,2,2,2]
# best_sp_list = [2,2,2,2]
# thresh_persent_s = [1,22,3]

# for thresh_persent in thresh_persent_s:
#     acc = np.mean(best_acc_list)*100
#     acc_std = np.std(best_acc_list)*100
#     with open('./results/readout_JK.txt', "a+", encoding='utf-8') as rf:
#         rf.write(
#             'readout %sJK-pooling 10 fold test average: %.2f±%.2f, sensitivity: %.2f±%.2f, specificity: %.2f±%.2f \n' % (
#                 a,
#                 np.mean(best_acc_list)*100, np.std(best_acc_list)*100,
#                 np.mean(best_sen_list)*100, np.std(best_sen_list)*100,
#                 np.mean(best_sp_list)*100, np.std(best_sp_list)*100
#             )
#         )

# import scipy


# data = scipy.io.loadmat("./RegionSeries.mat")
# print(data)
# print(np.shape(data['RegionSeries']))


# with open("/media/zhangke/data_disk/Topology_data/zhongda_xinxiang_data/zhongda_data_fmri/HC/HC_001/calStaticRes.pkl", "rb") as f:
#     data =  pickle.load(f)

#     print(data.source_mat.shape)

# d = {'batch_norm': True, 'batch_size': 128, 'beta': 1.0, 'dataset': 'NCI1', 'dim_hidden': 64, 'dropout': 0, 'epochs': 100, 'lap_dim': 8, 'lappe': True, 'layer_norm': False, 'lr': 0.001, 'nb_heads': 8, 'nb_layers': 7, 'normalization': 'sym', 'outdir': '', 'p': 1, 'pos_enc': None, 'save_logs': False, 'seed': 0, 'use_cuda': True, 'warmup': 2000, 'weight_decay': 1e-05, 'zero_diag': False}
# s = ""

# for key in d.keys():
#     s += "--{0} {1} ".format(key, d[key])

# print(s)

from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.datasets import TUDataset 
import torch
dataset = TUDataset(os.path.join("./TUDataset", 'IMDB-MULTI'), name='IMDB-MULTI')

class_weight = compute_class_weight('balanced', np.unique(dataset.data.y.numpy()),dataset.data.y.numpy())

print(class_weight)
print(torch.tensor(class_weight))
p_count = dataset.data.y.sum().float()
i = 0
for y_s in dataset.data.y:
    if y_s == 1:
        i+=1
print(p_count)
print(i)



