################################
# Convolutional models
################################


import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch_geometric.nn import global_add_pool, GINConv
from torch.nn.init import xavier_uniform_
import torch
import numpy as np
from utils import condition_number

class GCN_graph_classification(nn.Module):    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_graph_classification, self).__init__()


        # self.conv_layers = nn.ModuleList(
        #     [GraphConvolution(input_dim, hidden_dim)] +
        #     [GraphConvolution(hidden_dim, hidden_dim) for _ in range(1, num_layers-1)] + 
        #     [GraphConvolution(hidden_dim, output_dim)]
        # )
        self.conv_layers = nn.ModuleList(
            [GraphConvolution(input_dim, hidden_dim)] +
            [GraphConvolution(hidden_dim, hidden_dim) for _ in range(1, num_layers)]        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

        for param in ['p', 'q', 'r', 'c1', 'c2', 'c3', 'd1']:
            if param == 'd1':
                self.register_parameter(param, nn.Parameter(torch.zeros([1])))
            else:
                self.register_parameter(param, nn.Parameter(torch.zeros([1])))

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.p.data)
        xavier_uniform_(self.q.data)
        xavier_uniform_(self.r.data)
        xavier_uniform_(self.c1.data)
        xavier_uniform_(self.c2.data)
        xavier_uniform_(self.c3.data)
        xavier_uniform_(self.d1.data)

    def compute_generalized_laplacian(self, adj):

        # add weighted  self-connections to adjacency matrix 
        temp_adj = adj + self.d1 * torch.eye(adj.shape[0]).to(self.p.device)
        diags = temp_adj.sum(1)
        n = adj.shape[0]
        identity = torch.eye(n).to(self.p.device)


        p_diags = torch.zeros([n,n]).to(self.p.device)
        q_diags = torch.zeros([n,n]).to(self.p.device)
        r_diags = torch.zeros([n,n]).to(self.p.device)

        ind = np.diag_indices(n)
        p_diags[ind[0], ind[1]] = diags.clone() ** self.p 
        q_diags[ind[0], ind[1]] = diags.clone() ** self.q
        r_diags[ind[0], ind[1]] = diags.clone() ** self.r        
        gen_adj = self.c3 * p_diags - self.c2 * (q_diags.mm(temp_adj)).mm(r_diags) + self.c1 * identity
        # gen_adj = self.c3 * p_diags - (q_diags.mm(temp_adj)).mm(r_diags) + self.c1 * identity

        return gen_adj

    def forward(self, x, adj, batch):
        gen_adj = self.compute_generalized_laplacian(adj) 
        # gen_adj = adj + torch.eye(adj.shape[0]).to(self.p.device)
        h = x
        for i, layer in enumerate(self.conv_layers):
            h = layer(h, gen_adj)
            h = F.dropout(h, p=0.5, training = self.training)

            h = F.relu(h)

        out = global_add_pool(h, batch)
        out = self.fc(out)
        out = F.dropout(out, p=0.5, training = self.training)
        return F.log_softmax(out, dim=-1)
    





class GCN_node_classification(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout,adj, genlap=True, genlap_init='GCN'):
        super(GCN_node_classification, self).__init__()

        self.conv_layers = nn.ModuleList(
            [GraphConvolution(input_dim, hidden_dim)] +
            [GraphConvolution(hidden_dim, hidden_dim) for _ in range(1, num_layers-1)] + 
            [GraphConvolution(hidden_dim, output_dim)]
        )

        self.dropout = dropout
        self.genlap = genlap
        self.genlap_init = genlap_init

        # GCN initialization
        if genlap:
            if genlap_init == 'GCN':
                self.register_parameter('p', nn.Parameter(torch.zeros([1])))
                self.register_parameter('q', nn.Parameter(torch.zeros([1])-1/2))
                self.register_parameter('r', nn.Parameter(torch.zeros([1])-1/2))
                self.register_parameter('c1', nn.Parameter(torch.zeros([1])))
                self.register_parameter('c2', nn.Parameter(torch.ones([1])))
                self.register_parameter('c3', nn.Parameter(torch.zeros([1])))
                # self.register_parameter('d1', nn.Parameter(torch.ones([1])))
                self.register_parameter('d1', nn.Parameter(torch.ones(adj.shape[0])))

            elif genlap_init == 'all_zeros':
                for param in ['p', 'q', 'r', 'c1','c2', 'c3']:
                    self.register_parameter(param, nn.Parameter(torch.zeros([1])))
                self.register_parameter('d1', nn.Parameter(torch.zeros(adj.shape[0])))

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

    def compute_generalized_laplacian(self, adj):

        # add weighted  self-connections to adjacency matrix 
        temp_adj = adj + self.d1 * torch.eye(adj.shape[0]).to(self.p.device)
        diags = temp_adj.sum(1)
        n = adj.shape[0]
        identity = torch.eye(n).to(self.p.device)

        p_diags = torch.zeros([n,n]).to(self.p.device)
        q_diags = torch.zeros([n,n]).to(self.p.device)
        r_diags = torch.zeros([n,n]).to(self.p.device)

        ind = np.diag_indices(n)
        p_diags[ind[0], ind[1]] = diags.clone() ** self.p
        q_diags[ind[0], ind[1]] = diags.clone() ** self.q
        r_diags[ind[0], ind[1]] = diags.clone() ** self.r        
        gen_adj = self.c1 * p_diags + self.c2 * (q_diags.mm(temp_adj)).mm(r_diags) + self.c3 * identity

        return gen_adj

    def forward(self, x, adj):
        if self.genlap:
            gso = self.compute_generalized_laplacian(adj)
        else:
            gso = adj

        h = x
        for i, layer in enumerate(self.conv_layers):
            h = layer(h, gso)
            if i < len(self.conv_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, training=self.training)

        return F.log_softmax(h, dim=1)
    
