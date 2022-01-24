import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GINConv
from torch_geometric.utils import remove_self_loops, degree

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



# class gen_GINConv(GINConv):

#     def __init__(self, nn, eps=0, train_eps=False, **kwargs):
#         super(gen_GINConv,self).__init__(nn, eps=eps, train_eps=train_eps, **kwargs)
#         for param in ['p', 'q', 'r', 'c1', 'c2', 'c3', 'd1']:
#                 if param in ['c1','c2']:
#                     self.register_parameter(param, Parameter(torch.ones([1])))
#                 else:
#                     self.register_parameter(param, Parameter(torch.zeros([1])))



#     def forward(self, x, edge_index):
#         x = x.unsqueeze(-1) if x.dim() == 1 else x
#         edge_index, _ = remove_self_loops(edge_index)
#         degree_i = degree(edge_index[0],  num_nodes= x.shape[0])
#         degree_j = degree(edge_index[1],  num_nodes= x.shape[0])
#         edge_weight = (degree_i[edge_index[0]] ** self.q) * (degree_j[edge_index[1]] ** self.r)
    
#         out =  (self.c3 * (degree_i ** self.p) + self.c1).view(-1,1) * x
#         out += self.c2 * self.propagate(edge_index, x=x, edge_weight=edge_weight)
#         out =  self.nn(out)
#         return out

#     def message(self, x_j, edge_weight):
#         return edge_weight.view(-1, 1) * x_j


class gen_GINConv(GINConv):

    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(gen_GINConv,self).__init__(nn, eps=eps, train_eps=train_eps, **kwargs)
        # for param in ['p', 'q', 'r', 'c1', 'c2', 'c3', 'd1']:
        #         if param in ['c1','c2']:
        #             self.register_parameter(param, Parameter(torch.ones([1])))
        #         else:
        #             self.register_parameter(param, Parameter(torch.zeros([1])))



    def forward(self, x, edge_index, genlap_params):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, edge_weight=genlap_params['d1'])
        # print(edge_index)
        # sadsdsa
        degree_i = degree(edge_index[0],  num_nodes= x.shape[0]) + genlap_params['d1']
        degree_j = degree(edge_index[1],  num_nodes= x.shape[0]) + genlap_params['d1']
        edge_weight = (degree_i[edge_index[0]] ** genlap_params['q']) * (degree_j[edge_index[1]] ** genlap_params['r'])
    
        out =  (genlap_params['c1'] * (degree_i ** genlap_params['p']) + genlap_params['c3']).view(-1,1) * x
        out += genlap_params['c2'] * self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out =  self.nn(out)
        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
