import numpy as np
import torch
import pickle
import time
import matplotlib.pyplot as plt
import scipy.sparse
import networkx as nx
def shuffle(W,c):
    # relabel the vertices at random
    idx=np.random.permutation( W.shape[0] )
    #idx2=np.argsort(idx) # for index ordering wrt classes
    W_new=W[idx,:]
    W_new=W_new[:,idx]
    c_new=c[idx]
    return W_new , c_new , idx 

def block_model(c,p,q):
    n=len(c)
    W=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if c[i]==c[j]:
                prob=p
            else:
                prob=q
            if np.random.binomial(1,prob)==1:
                W[i,j]=1
                W[j,i]=1     
    return W


def unbalanced_block_model(nb_of_clust, clust_size_min, clust_size_max, p, q):  
    c = []
    for r in range(nb_of_clust):
        if clust_size_max==clust_size_min:
            clust_size_r = clust_size_max
        else:
            clust_size_r = np.random.randint(clust_size_min,clust_size_max,size=1)[0]
        val_r = np.repeat(r,clust_size_r,axis=0)
        c.append(val_r)
    c = np.concatenate(c)  
    W = block_model(c,p,q)  
    return W,c



class generate_SBM_graph():

    def __init__(self, SBM_parameters): 

        # parameters
        nb_of_clust = SBM_parameters['nb_clusters']
        clust_size_min = SBM_parameters['size_min']
        clust_size_max = SBM_parameters['size_max']
        p = SBM_parameters['p']
        q = SBM_parameters['q']

        # block model
        W, c = unbalanced_block_model(nb_of_clust, clust_size_min, clust_size_max, p, q)
        
        # shuffle
        W, c, idx = shuffle(W,c)
        train_ratio, val_ratio, test_ratio = 0.8,0.1,0.1
        # signal on block model
        u = np.zeros(c.shape[0])

        train_idx, val_idx , test_idx = [], [], []
        for r in range(nb_of_clust):
            cluster = np.where(c==r)[0]
            rand_cluster = np.random.permutation(cluster)
            cl_splits = np.split(rand_cluster, [int(train_ratio*len(cluster)),int(train_ratio*len(cluster))+ int(val_ratio*len(cluster))])
            train_idx.append(cl_splits[0])
            val_idx.append(cl_splits[1])
            test_idx.append(cl_splits[2])
            s = cluster[np.random.randint(cluster.shape[0])]
            # no change on attribute vector , as we want uninformative attributes
            u[s] = r+1
            

        # target
        target = c
        u = target
        
        # convert to pytorch
        Wnum = W
        W = torch.from_numpy(W)
        idx = torch.from_numpy(idx) 
        u = torch.from_numpy(u) 
        u = u.to(torch.float32)                      
        target = torch.from_numpy(target)
        
        # attributes
        self.total_nodes = W.size(0)
        self.W = W
        self.rand_idx = idx
        self.node_feat = u
        self.node_label = target
        self.train_idx = torch.from_numpy(np.concatenate(train_idx).ravel())
        self.val_idx = torch.from_numpy(np.concatenate(val_idx).ravel())
        self.test_idx = torch.from_numpy(np.concatenate(test_idx).ravel())
    
        # transforms integer indices to boolean masks
        # self.train_mask, self.val_mask, self.test_mask = torch.zeros(self.nb_nodes), torch.zeros(self.nb_nodes),torch.zeros(self.nb_nodes)
        self.train_mask, self.val_mask, self.test_mask = torch.zeros(self.total_nodes, dtype= torch.bool), torch.zeros(self.total_nodes, dtype= torch.bool),torch.zeros(self.total_nodes, dtype= torch.bool)
        self.train_mask[self.train_idx] = 1
        self.val_mask[self.val_idx] = 1
        self.test_mask[self.test_idx] = 1
        
        #Plot Adj matrix
        
        idx = np.argsort(self.rand_idx) 
        self.sort_idx = idx
        # W = self.W
        # W2 = W[idx,:]
        # W2 = W2[:,idx]

        # plt.show()

        # g = nx.from_numpy_matrix(Wnum)
        # nx.draw_networkx(g, labels = {k:v for k,v in enumerate(target)})
        # plt.show()
        # sdasdsa
# Generate and save SBM graphs
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data

def generate_graphs(nb_nodes, nb_graphs, p , q):
    dataset = []
    # configuration   
    SBM_parameters = {}
    SBM_parameters['nb_clusters'] = 3
    SBM_parameters['size_min'] = nb_nodes
    SBM_parameters['size_max'] = nb_nodes 
    SBM_parameters['p'] = p 
    SBM_parameters['q'] = q
    SBM_parameters['pq_ratio'] = SBM_parameters['p']/SBM_parameters['q']  

    W_list,sort_idx_list, p_list, q_list, pplot_list, qplot_list = [], [], [], [], [], []
    p_previous, q_previous, pq  = 0.5, 0.25, 2
    for i in range(nb_graphs):
        # print(f'{SBM_parameters['p']}')
        print(SBM_parameters['q'])
        print(f'Graph processed: {i}')
        p_list.append(SBM_parameters['p'])
        pplot_list.append(p_previous)
        qplot_list.append(q_previous)
        q_list.append(SBM_parameters['q'])
        data = generate_SBM_graph(SBM_parameters)
        W_list.append(data.W)
        sort_idx_list.append(data.sort_idx)


        graph = Data(
            x = data.node_feat,
            edge_index = dense_to_sparse(data.W)[0],
            y = data.node_label,
            train_mask = data.train_mask,
            val_mask = data.val_mask,
            test_mask = data.test_mask,
            sparsity_level = i
        )
        dataset.append(graph)
        
        # change sparsity level and retain pq ratio constant 
        
        q_new = SBM_parameters['q'] * 0.85
        p_new = q_new * SBM_parameters['pq_ratio']

        # step = 0.02
        # p_new = SBM_parameters['p'] - step
        # q_new = SBM_parameters['q'] - (step / SBM_parameters['pq_ratio'])
        SBM_parameters['p']  = p_new
        SBM_parameters['q']  = q_new

        if i == 8:
            step = 0.04
        else:
            step = 0.02
        p_new2 = p_previous - step
        q_new2 = q_previous - (step / pq)
        p_previous  = p_new2
        q_previous  = q_new2

    fig, ax = plt.subplots(3,5)
    for i in range(nb_graphs):
        W2 = W_list[i][sort_idx_list[i],:]
        W2 = W2[:,sort_idx_list[i]]
        ax[int(i/5),i%5].spy(W2,precision=0.01, markersize=1)
        # ax[int(i/5),i%5].axis('off')
        ax[int(i/5),i%5].set_yticklabels([])
        ax[int(i/5),i%5].set_xticklabels([])
        ax[int(i/5),i%5].set_xlabel(f'p= {pplot_list[i]:0.2f}, q= {qplot_list[i]:0.2f}', fontsize = 10)
    plt.tight_layout()
    plt.show()    
    # plt.savefig('./figures/graphs_20nodes_15graphs_sparsity_levels.png')    
    return dataset


class SBM_pyg(InMemoryDataset):
    def __init__(self, root, nb_nodes=20, nb_graphs=20, p=0.85, q=0.15, transform=None, pre_transform=None):
        self.nb_nodes = nb_nodes
        self.nb_graphs = nb_graphs
        self.p, self.q = p, q
        self.root = root + '_' + str(self.nb_nodes) + 'nodes_' + str(self.nb_graphs)+'graphs_'+ str(int(p*100)) + '_' + str(int(q*100))
        print(self.root)
        super(SBM_pyg, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['tentative']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = generate_graphs(self.nb_nodes, self.nb_graphs, self.p, self.q)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# a = SBM_pyg('./data/SBM_final', nb_nodes=100, nb_graphs=15, p = 0.85, q=0.25)