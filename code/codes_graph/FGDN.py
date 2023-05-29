import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import ChebConv

class FGDN(nn.Module):
    def __init__(self,in_size, nb_class, d_model, dropout=0.1, nb_layers=4):
        super(FGDN, self).__init__()
        #self.args = args
        self.features = in_size
        self.hidden_dim = d_model
        self.num_layers = nb_layers
        self.num_classes = nb_class
        self.dropout =dropout

        self.conv1 = ChebConv(self.features, self.hidden_dim,K=3)
        self.prelu_a1 = nn.Parameter(torch.Tensor([0.25]))
        self.prelu_a2 = nn.Parameter(torch.Tensor([0.25]))
        self.prelu_a3 = nn.Parameter(torch.Tensor([0.25]))

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(ChebConv(self.hidden_dim, self.hidden_dim,K=3))

        #self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.fc2 = nn.Linear(self.hidden_dim, 1)

        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.prelu(self.fc1(x),self.prelu_a3)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index,data.batch
        x = F.prelu(self.conv1(x, edge_index),self.prelu_a1)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index),self.prelu_a2)
        x = global_add_pool(x, batch)
        x = self.fc_forward(x)
        #x = F.log_softmax(x, dim=-1)
        x = F.log_softmax(x, dim=-1)
        return x

    def __repr__(self):
        return self.__class__.__name__
