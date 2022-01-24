import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.utils import to_dense_adj, to_undirected
import torch.nn.functional as F
from sbm_dataset import SBM_pyg


def encode_onehot(labels):
    classes = set(labels)
    # print(f'labels: {labels}')
    # print(f'classes: {classes}')
    classes_dict = {c.item(): np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # print(f'classes_dict: {classes_dict}')
    # print(list(map(classes_dict.get, labels)))
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)


    return labels_onehot


def load_data_gs(path="./data/", dataset="MUTAG", device=None):
    print('Loading {} dataset...'.format(dataset))
    pre_transform = NormalizeFeatures()
    data = TUDataset(path, dataset, pre_transform=pre_transform)[0].to(device)
    print(data)
    features, labels, edges, batch = data.x, data.y, data.edge_index, data.batch
    # adj contains all information from all graphs    
    adj = to_dense_adj(to_undirected(edges)).squeeze()

    return adj, features, labels, batch



def load_data(path="./data/", dataset_name="Cora", nb_nodes=20, nb_graphs=20, p =None, q=None, device=None):
    print('Loading {} dataset...'.format(dataset_name))

    pre_transform = NormalizeFeatures()

    if dataset_name == "SBM":
        dataset = SBM_pyg('./data/SBM', nb_nodes=nb_nodes, nb_graphs= nb_graphs, p = p, q= q,  pre_transform=None)
        features, labels, adj = [], [], []
        idx_train, idx_val, idx_test = [], [], []
        for exp in range(len(dataset)):
            data = dataset[exp]
            features.append(data.x.view(-1,1))
            labels.append(data.y)
            adj.append(to_dense_adj(to_undirected(data.edge_index)).squeeze())
            idx_train.append(data.train_mask)
            idx_val.append(data.val_mask)
            idx_test.append(data.test_mask)

    elif dataset_name in {"Cora", "CiteSeer", "PubMed"}:
        data = Planetoid(path, dataset_name, pre_transform=pre_transform)[0].to(device)
        features, labels, edges = data.x, data.y, data.edge_index
        adj = to_dense_adj(to_undirected(edges)).squeeze()
        idx_train = data.train_mask
        idx_val = data.val_mask
        idx_test = data.test_mask

    else:
        print("Not a correct dataset name!")
        exit()


    return adj, features, labels, idx_train, idx_val, idx_test
    


def load_data_old(path="./data/", dataset="cora", device=None):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # path = osp.join(path,dataset)
    idx_features_labels = np.genfromtxt("{}/{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)

    # adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = adj + sp.eye(adj.shape[0])
    # A = adj + sp.eye(adj.shape[0])
    # A = adj
    # (n, n) = A.shape
    # diags = A.sum(axis=1).flatten()
    # p, q = 0.0,-1
    # p_diags, q_diags = diags.astype(float), diags.astype(float) 
    # p_diags[0, np.diag_indices(n)] = np.float_power(p_diags, p)   
    # q_diags[0, np.diag_indices(n)] = np.float_power(q_diags, q) 
    # pD = sp.spdiags(p_diags, [0], n, n, format='csr')
    # qD = sp.spdiags(q_diags, [0], n, n, format='csr')
    # generalized_laplacian = pD + qD.dot(A)
    # generalized_laplacian = pD - qD.dot(A)
    # adj = generalized_laplacian
    # adj = normalize(generalized_laplacian)
    # adj = sparse_mx_to_torch_sparse_tensor(generalized_laplacian)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(np.where(labels)[1]).to(device)

    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

    # # make generalized laplacian 
    # temp_adj = adj + torch.eye(adj.shape[0]).to(device)
    # diags = temp_adj.sum(1)
    # n = adj.shape[0]
    # identity = torch.eye(n).to(device)


    # p_diags = torch.zeros([n,n]).to(device)
    # q_diags = torch.zeros([n,n]).to(device)
    # r_diags = torch.zeros([n,n]).to(device)

    # ind = np.diag_indices(n)
    # p_diags[ind[0], ind[1]] = diags.clone() ** (-1.2456) 
    # q_diags[ind[0], ind[1]] = diags.clone() ** (0.018)
    # r_diags[ind[0], ind[1]] = diags.clone() ** (0.24)        
    # # gen_adj = p_diags - self.s*(q_diags.mm(temp_adj)).mm(r_diags)
    # gen_adj = p_diags - (q_diags.mm(temp_adj)).mm(r_diags) + (-3.95) * identity

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    # return gen_adj, features, labels, idx_train, idx_val, idx_test

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    sort_idx = np.argsort(labels)
    # print(f'sorted p:{preds[sort_idx]}')
    # print(f'sorted l:{labels[sort_idx]}')
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()


def condition_number(mx):
    _, spectral_norm, _ = sp.linalg.svds(mx.detach().numpy(),k=1)
    _, spectral_norm_inv, _ = sp.linalg.svds(torch.inverse(mx).detach().numpy(),k=1)
    # print(f'condition number: {spectral_norm * spectral_norm_inv}')
    # return torch.norm(mx, 2) * torch.norm(torch.inverse(mx), 2)
    return spectral_norm * spectral_norm_inv