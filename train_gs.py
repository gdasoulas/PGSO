from __future__ import division
from __future__ import print_function
from datetime import datetime
from pathlib import Path
from plot_utils import plot_accs_per_initialization, plot_scalar_params_per_initialization

import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, load_data_old, accuracy
import models.mp as mp_models
from torch_geometric.utils import to_dense_adj, to_undirected
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures, OneHotDegree
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/',
                    help='Directory of datasets; default is ./data/')
parser.add_argument('--dataset', type=str, default='MUTAG',
                    help='Dataset name; default is Mutag')
parser.add_argument('--model', type=str, default='GIN',
                    help='Model to be trained; default is GIN')                    
parser.add_argument('--device', type=int, default=-1,
                    help='Set CUDA device number; if set to -1, disables cuda.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Size of batch.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lr_patience', type=float, default=50,
                    help='Number of epochs waiting for the next lr decay.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of aggregation layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--outdir', type=str, default='./exps/',
                    help='Directory of experiments output; default is ./exps/')

args = parser.parse_args()


expfolder = osp.join(args.outdir, args.dataset)
expfolder = osp.join(expfolder,datetime.now().strftime("%Y%m%d_%H%M%S"))
Path(expfolder).mkdir(parents=True, exist_ok=True)


device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)



# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.device > 0:
#     torch.cuda.manual_seed(args.seed)

# pre_transform = NormalizeFeatures()
pre_transform = None
if args.dataset.startswith('IMDB'):
    pre_transform = OneHotDegree(max_degree=150)
dataset = TUDataset(args.datadir, name=args.dataset, pre_transform=pre_transform).shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

print(dataset.data)
# Model and optimizer
for genlap_init in ['GCN', 'all_zeros', 'SymLaplacian', 'RWLaplacian', 'Adjacency']:

    if 'GIN' in args.model:
        model = getattr(mp_models, args.model)(input_dim=dataset.num_features,
                    hidden_dim=args.hidden,
                    output_dim=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout, genlap_init = genlap_init).to(device)
    elif 'GCN' in args.model:
        model = getattr(conv_models, args.model)(input_dim=dataset.num_features,
                    hidden_dim=args.hidden,
                    output_dim=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout, adj=adj).to(device)
    else:
        print('No such model!')
        exit()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)
    exp_param_list = ['gnn_layers.'+ str(i)+ '.p' for i in range(1,args.num_layers+1)]
    exp_param_list += ['gnn_layers.'+ str(i)+ '.q' for i in range(1,args.num_layers+1)]
    exp_param_list += ['gnn_layers.'+ str(i)+ '.r' for i in range(1,args.num_layers+1)]

    exp_params = list(filter(lambda kv: kv[0] in exp_param_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in exp_param_list, model.named_parameters()))
    exp_params = [param[1] for param in exp_params]
    base_params = [param[1] for param in base_params]

    optimizer = optim.Adam([
                            {'params': base_params, 'lr':args.lr},
                            {'params': exp_params, 'lr': 0.05}
                            ], lr=args.lr, weight_decay=args.weight_decay)

    # optimizer = optim.Adam(model.parameters(),
    #                        lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = None
    if args.lr_patience > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_patience, gamma=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.lr_patience, verbose=True)


    def train(epoch):
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            # adj = to_dense_adj(to_undirected(data.edge_index)).squeeze()
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_dataset)


    def test(loader):
        model.eval()
        correct = 0
        for data in loader:
            data = data.to(device)
            # adj = to_dense_adj(to_undirected(data.edge_index)).squeeze()
            output = model(data.x, data.edge_index, data.batch)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)


    # Train model
    t_total = time.time()
    states = []
    for epoch in range(1,args.epochs+1):
        train_loss = train(epoch)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        whole_state = {
            'epoch': epoch,
            'model_state_dict': {key:val.clone() for key,val in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': train_loss,
            'acc_train': train_acc,
            'acc_val': test_acc,
            } 

        states.append(whole_state)    
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                        train_acc, test_acc))

        if args.lr_patience > 0:
            lr_scheduler.step()

    torch.save(states, osp.join(expfolder,f'model_states_init={genlap_init}.pth'))

plot_scalar_params_per_initialization(expfolder)
plot_accs_per_initialization(expfolder)