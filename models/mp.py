################################
# Message passing models
################################
from layers import *
import torch.nn as nn
from torch_geometric.nn import global_add_pool, GINConv
import torch.nn.functional as F

class GIN_graph_classification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, genlap_use, genlap_init):
        super(GIN_graph_classification, self).__init__()
        self.genlap_use= genlap_use
        # GIN layers
        if self.genlap_use:
            self.gnn_layers = nn.ModuleList([gen_GINConv(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, hidden_dim)))
            ])
        else:
            self.gnn_layers = nn.ModuleList([GINConv(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, hidden_dim)))
            ])

        for _ in range(1, num_layers):
            if self.genlap_use:    
                gnn = gen_GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, hidden_dim)))
            else:
                gnn = GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)))
            self.gnn_layers.append(gnn)


        # GCN initialization
        if genlap_use:
            if genlap_init == 'GCN':
                self.register_parameter('p', nn.Parameter(torch.zeros([1])))
                self.register_parameter('q', nn.Parameter(torch.zeros([1])-1/2))
                self.register_parameter('r', nn.Parameter(torch.zeros([1])-1/2))
                self.register_parameter('c1', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c2', nn.Parameter(torch.ones([1])))
                self.register_parameter('c3', nn.Parameter(torch.zeros([1])))
                self.register_parameter('d1', nn.Parameter(torch.ones([1])))
            
            elif genlap_init == 'GIN':
                self.register_parameter('p', nn.Parameter(torch.zeros([1])))
                self.register_parameter('q', nn.Parameter(torch.zeros([1])-1))
                self.register_parameter('r', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c1', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c2', nn.Parameter(torch.ones([1])))
                self.register_parameter('c3', nn.Parameter(torch.zeros([1])))
                self.register_parameter('d1', nn.Parameter(torch.zeros([1])))
            
            elif genlap_init == 'all_zeros':
                for param in ['p', 'q', 'r', 'c1','c2', 'c3', 'd1']:
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))

            elif genlap_init == 'Laplacian':
                self.register_parameter('p', nn.Parameter(torch.ones([1])))
                self.register_parameter('q', nn.Parameter(torch.zeros([1])))
                self.register_parameter('r', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c1', nn.Parameter(torch.ones([1])))
                self.register_parameter('c2', nn.Parameter(-torch.ones([1])))
                self.register_parameter('c3', nn.Parameter(torch.zeros([1])))
                self.register_parameter('d1', nn.Parameter(torch.zeros([1])))   

            elif genlap_init == 'SymLaplacian':
                self.register_parameter('p', nn.Parameter(torch.zeros([1])))
                self.register_parameter('q', nn.Parameter(torch.zeros([1])-1/2))
                self.register_parameter('r', nn.Parameter(torch.zeros([1])-1/2))
                self.register_parameter('c1', nn.Parameter(torch.ones([1])))
                self.register_parameter('c2', nn.Parameter(-torch.ones([1])))
                self.register_parameter('c3', nn.Parameter(torch.zeros([1])))
                self.register_parameter('d1', nn.Parameter(torch.zeros([1])))   
            
            elif genlap_init == 'RWLaplacian':
                self.register_parameter('p', nn.Parameter(torch.zeros([1])))
                self.register_parameter('q', nn.Parameter(torch.zeros([1])-1))
                self.register_parameter('r', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c1', nn.Parameter(torch.ones([1])))
                self.register_parameter('c2', nn.Parameter(-torch.ones([1])))
                self.register_parameter('c3', nn.Parameter(torch.zeros([1])))
                self.register_parameter('d1', nn.Parameter(torch.zeros([1])))   

            elif genlap_init == 'Adjacency':
                self.register_parameter('p', nn.Parameter(torch.zeros([1])))
                self.register_parameter('q', nn.Parameter(torch.zeros([1])))
                self.register_parameter('r', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c1', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c2', nn.Parameter(torch.ones([1])))
                self.register_parameter('c3', nn.Parameter(torch.zeros([1])))
                self.register_parameter('d1', nn.Parameter(torch.zeros([1])))                   

            self.genlap_params = {'p':self.p,'q':self.q,'r':self.r,'c1':self.c1,'c2':self.c2,'c3':self.c3,'d1':self.d1}



        # batch normalization layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        for gnn_layer, bn_layer in zip(self.gnn_layers, self.bn_layers):
            if self.genlap_use:
                x = F.relu(gnn_layer(x, edge_index, self.genlap_params))
            else:
                x = F.relu(gnn_layer(x, edge_index))
            x = bn_layer(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)     
        return F.log_softmax(x, dim=-1)
  
# class GIN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
#         super(GIN, self).__init__()
        
#         # GIN layers
#         self.gnn_layers = nn.ModuleList([GINConv(nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim), 
#                 nn.ReLU(), 
#                 nn.Linear(hidden_dim, hidden_dim)))
#         ])
#         for _ in range(1, num_layers):
#             gnn = GINConv(nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim), 
#                 nn.ReLU(), 
#                 nn.Linear(hidden_dim, hidden_dim)))
#             self.gnn_layers.append(gnn)

#         # batch normalization layers
#         self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, edge_index, batch):
#         for gnn_layer, bn_layer in zip(self.gnn_layers, self.bn_layers):
#             x = F.relu(gnn_layer(x, edge_index))
#             x = bn_layer(x)

#         x = global_add_pool(x, batch)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.fc2(x)     
#         return F.log_softmax(x, dim=-1)


