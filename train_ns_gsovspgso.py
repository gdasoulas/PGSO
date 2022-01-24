from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, load_data_old, accuracy, condition_number
from models.conv import GCN_node_classification, GCN_graph_classification
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from plot_utils import plot_accs_per_use


########################################################################################
# Train and test functions 
########################################################################################

def train(epoch):
    t = time.time()
    model.train()

    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = criterion(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    # cond_values.append(condition_number(model.gen_adj).item())

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t)
        #   'cond: {:.1f}'.format(condition_number(model.gen_adj))
          )

    whole_state = {
        'epoch': epoch,
        'model_state_dict': {key:val.clone() for key,val in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': loss_train,
        'loss_val': loss_val,
        'acc_train': acc_train,
        'acc_val': acc_val,
        }
    return whole_state


def test():
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


########################################################################################
# Parse arguments 
########################################################################################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, default='./data/',
                    help='Directory of datasets; default is ./data/')
parser.add_argument('--outdir', type=str, default='./exps/',
                    help='Directory of experiments output; default is ./exps/')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Dataset name; default is Cora')
parser.add_argument('--device', type=int, default=-1,
                    help='Set CUDA device number; if set to -1, disables cuda.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--exp_lr', type=float, default=0.01,
                    help='Initial learning rate for exponential parameters.')
parser.add_argument('--lr_patience', type=float, default=50,
                    help='Number of epochs waiting for the next lr decay.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--genlap', action='store_true', default=False,
                    help='Utilization of GenLap')

args = parser.parse_args()
device = torch.device('cuda:'+str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device > 0:
    torch.cuda.manual_seed(args.seed)


expfolder = osp.join(args.outdir, args.dataset+'_gsovspgso')
expfolder = osp.join(expfolder,datetime.now().strftime("%Y%m%d_%H%M%S"))
Path(expfolder).mkdir(parents=True, exist_ok=True)

########################################################################################
# Data loading and model setup 
########################################################################################

if args.dataset == 'cora':
    adj, features, labels, idx_train, idx_val, idx_test = load_data_old('./data/cora_old', args.dataset, device)
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.datadir, args.dataset, device)

torch.save(adj, osp.join(expfolder,f'adj.pth'))


# for genlap_init in ['GCN', 'all_zeros', 'SymLaplacian', 'RWLaplacian', 'Adjacency']:
for genlap_use in [True,False]:

    print(genlap_use)
    # Model and optimizer
    model = GCN_node_classification(input_dim=features.shape[1],
                hidden_dim=args.hidden,
                output_dim=labels.max().item() + 1,
                num_layers=args.num_layers,
                dropout=args.dropout,adj=adj, genlap=genlap_use, genlap_init='GCN').to(device)


    # Exponential parameters have a different learning rate than other multiplicative parameters
    exp_param_list = ['p', 'q', 'r']
    exp_params = list(filter(lambda kv: kv[0] in exp_param_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in exp_param_list, model.named_parameters()))
    exp_params = [param[1] for param in exp_params]
    base_params = [param[1] for param in base_params]

    optimizer = optim.Adam([
                            {'params': base_params, 'lr':args.lr},
                            {'params': exp_params, 'lr': args.exp_lr}
                            ], lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = None
    if args.lr_patience > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_patience, gamma=0.6)

    cond_values = []

    # Train model
    t_total = time.time()
    p_values, q_values, r_values, c1_values, c2_values, c3_values, d1_values = [], [], [], [], [], [], []



    states = []
    for epoch in range(1,args.epochs+1):
        state = train(epoch)
        states.append(state)

        if args.lr_patience > 0:
            lr_scheduler.step()
        
    torch.save(states, osp.join(expfolder,f'model_states_gso_use={genlap_use}.pth'))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()

plot_accs_per_use(expfolder)
