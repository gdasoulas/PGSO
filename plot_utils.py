import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os.path as osp
from utils import condition_number
import pandas as pd
from scipy.sparse.linalg import svds
import random 
import seaborn as sns

workspace = './exps'
outfolder = './figures'


def plot_scalar_params_per_initialization(outfolder):
    dataset = outfolder.split('/')[2]
    param_names = ['c1','c2','c3','p','q','r', 'd1']
    correct_params = {'p':r'$e_1$','q':r'$e_2$','r':r'$e_3$','c1':r'$m_1$','c2':r'$m_2$','c3':r'$m_3$', 'd1':r'$a$'}
    correct_init = {'RWLaplacian':'Random-Walk Laplacian','SymLaplacian':'Symmetric Laplacian','Adjacency':'Adjacency', 'GCN':'GCN operator','all_zeros':'All zeros'}
    data = []
    for init in ['GCN','Adjacency','all_zeros','RWLaplacian','SymLaplacian'] :

    # loading model states
        print(init)
        model_states = torch.load(osp.join(outfolder, f'model_states_init={init}.pth'), map_location=torch.device('cpu'))
      
        # model_states = torch.load(osp.join(workspace, filename))
        # print(model_states[0]['model_state_dict']['p'])
        params = {param: [] for param in param_names}
        for state in model_states:
            for param in param_names:
                params[param].append(state['model_state_dict'][param])
                data.append([correct_params[param], state['epoch'], state['model_state_dict'][param].item(), correct_init[init]])
        
    datafr = pd.DataFrame(data=data,columns=['Parameter','Epoch','Parameter Value','Initialization'])
        # sdsd
    plt.figure(figsize=(12,8))

    sns.set_theme(style="whitegrid",font_scale=2.5)
    g = sns.FacetGrid(data=datafr, col="Initialization", height=8.27)
    g.map(sns.scatterplot,"Epoch", "Parameter Value","Parameter", s=100, edgecolor=None)

    g.set(xlabel=r'Epochs')
    g.set(ylabel='PGSO Parameters')
    g.set_titles('{col_name}')
    lgnd = plt.legend(bbox_to_anchor = (-.65,1.23), ncol=7)
    for i in [0,1,2,3,4,5,6]:
        lgnd.legendHandles[i]._sizes = [300]
    print(dataset)
    g.savefig(dataset+'_init.pdf')


def plot_accs_per_initialization(outfolder):
    fig,ax = plt.subplots(1,5,figsize=(20,5))

    for ind, init in enumerate(['GCN','Adjacency','all_zeros','RWLaplacian','SymLaplacian']) :
        model_states = torch.load(osp.join(outfolder, f'model_states_init={init}.pth'), map_location=torch.device('cpu'))
        acc_train, acc_val = [], []

        for state in model_states:
            acc_train.append(state['acc_train'])
            acc_val.append(state['acc_val'])


        epochs = range(1, model_states[-1]['epoch']+1)  # last epoch  
        # plt.figure()
        ax[ind].plot(epochs, acc_train, label='Train Acc.')
        ax[ind].plot(epochs, acc_val, label='Validation Acc.')
        ax[ind].legend()
        ax[ind].set_title(f'Accuracies for init={init}')

        ax[ind].set_xlabel('Epochs')
        ax[ind].set_ylabel(f'Train/Val Accuracy')
    plt.tight_layout()
    plt.savefig(osp.join(outfolder, f'genlap_accs_all_inits.pdf'))


def plot_accs_per_use(outfolder):
    fig,ax = plt.subplots(1,3,figsize=(15,5))

    # for ind, init in enumerate(['GCN','Adjacency','all_zeros','RWLaplacian','SymLaplacian']) :
    model_states_true = torch.load(osp.join(outfolder, f'model_states_gso_use=True.pth'), map_location=torch.device('cpu'))
    model_states_false = torch.load(osp.join(outfolder, f'model_states_gso_use=False.pth'), map_location=torch.device('cpu'))
    acc_train_true, acc_val_true,acc_train_false, acc_val_false, = [], [], [], []
    loss_train_true, loss_val_true,loss_train_false, loss_val_false, = [], [], [], []
    for state in model_states_true:
        acc_train_true.append(state['acc_train'])
        acc_val_true.append(state['acc_val'])
        loss_train_true.append(state['loss_train'])
        # loss_val_true.append(state['loss_val'])   
    for state in model_states_false:
        acc_train_false.append(state['acc_train'])
        acc_val_false.append(state['acc_val'])
        loss_train_false.append(state['loss_train'])
        # loss_val_false.append(state['loss_val'])   

    epochs = range(1, model_states_true[-1]['epoch']+1)  # last epoch  

    # accuracies
    model_name = 'GIN' if 'ptc' in outfolder else 'GCN'
    ax[0].plot(epochs, acc_train_true, label=f'Train Accuracy of {model_name}-PGSO')
    ax[0].plot(epochs, acc_train_false, label=f'Train Accuracy of {model_name}')

    ax[0].legend(loc='lower right')
    ax[0].set_title(f'Train Accuracy with and without PGSO')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel(f'Train Accuracy')
    
    ax[1].plot(epochs, acc_val_true, label=f'Val. Accuracy of {model_name}-PGSO')
    ax[1].plot(epochs, acc_val_false, label=f'Val. Accuracy of {model_name}')

    ax[1].legend(loc='lower right')
    ax[1].set_title(f'Validation accuracy with and without PGSO')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel(f'Validation Accuracy')

    ax[2].plot(epochs, loss_train_true, label=f'Train Loss of {model_name}-PGSO')
    ax[2].plot(epochs, loss_train_false, label=f'Train Loss of {model_name}')

    ax[2].legend(loc='upper right')
    ax[2].set_title(f'Loss convergence with and without PGSO')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel(f'Loss')
    
    plt.tight_layout()

    plt.savefig(osp.join(outfolder, f'genlap_accs_use.pdf'))

# plot_accs_per_use('./exps/PTC_MR_gsovspgso/20201121_223622')
# plot_accs_per_use('./exps/cora_gsovspgso/20201121_212518')
# plot_scalar_params_per_initialization('./exps/cora/20201115_123203')
# plot_accs_per_initialization('./exps/cora/20201115_123203')
# plot_scalar_params_per_initialization('./exps/MUTAG/20201115_142907')
# plot_accs_per_initialization('./exps/MUTAG/20201115_142907')


def plot_scalar_params_per_sparsity_level(outfolder):
    # loading model states
    # outfolder = workspace + '/SBM_200nodes_40graphs_50_25'
    np.random.seed(4)
    param_names = ['c3','c2','c1','p','q','r', 'd1']
    correct_params = {'p':r'$e_1$','q':r'$e_2$','r':r'$e_3$','c1':r'$m_3$','c2':r'$m_2$','c3':r'$m_1$', 'd1':r'$a$'}
    prob_p = np.linspace(0.5,0.2,num=15)
    prob_p = list(range(1,16))

    params = {param: [] for param in param_names}
    variance = []
    nexps = 15

    data = []
    for exp in range(1,nexps+1):
        model_states = torch.load(osp.join(outfolder, f'model_states_exp={exp}.pth'))
        for param in param_names:

            params[param].append(model_states[-1]['model_state_dict'][param])
            val = model_states[-1]['model_state_dict'][param].item()
            data.append([correct_params[param], prob_p[exp-1], val])


    datafr = pd.DataFrame(data=data,columns=['Parameter','Sparsity Level','Parameter Value'])
    
    dfCopy = datafr.copy()
    plt.figure(figsize=(9,7))
    sns.set(rc={"lines.linewidth": 0.7})
    sns.set_theme(style="whitegrid",font_scale=1.6)
    # sns.set()
    g = sns.pointplot(x="Sparsity Level", y="Parameter Value", hue="Parameter", data=dfCopy,capsize=.15,errwidth=1.5, palette = 'deep')
    g.set(xlabel=r'Probability $p$')
    g.set(ylabel='PGSO Parameters')
    xlabels = [ "%.2f"%n for n in np.linspace(0.5,0.2,16)]
    print(xlabels)
    g.set(xticklabels = xlabels)
    g.set(ylim=[-2,4.0])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.savefig(osp.join(outfolder,'parameters_sparsity_levels_final.pdf'))

def plot_spectral_support(expfolder):
    # loading model states
    param_names = {'p','q','r','c1','c2','c3', 'd1'}
    params = {param: [] for param in param_names}
    model_states = torch.load(osp.join(expfolder, f'model_states.pth'))
    adj = torch.load(osp.join(expfolder, f'adj.pth'))
    degrees = adj.sum(0)
    bounds, condition_numbers, gcn_condition_numbers = [], [], []
    epochs = model_states[-1]['epoch']
    # epochs = 3
    
    font = {'family' : 'normal',
            'size'   : 10}
    import matplotlib
    matplotlib.rc('font', **font)
    plt.figure(figsize=(5,3))

    for e in range(epochs):
        print(f'Epoch: {e+1}')
        for param in param_names:
            params[param].append(model_states[e]['model_state_dict'][param])


        # compute Gershgorin disks
        tmp1 = degrees + params['d1'][-1]
        tmp2 = params['c3'][-1] * tmp1 ** params['p'][-1]
        tmp3 = params['c2'][-1] * tmp1 ** (params['q'][-1]+params['r'][-1])
        tmp4 = tmp3 * params['d1'][-1]
        center = tmp2 + tmp4 + params['c1'][-1]
        radius = np.abs(params['c2'][-1])*(tmp1)**(params['q'][-1]+params['r'][-1])*degrees

        min_bound = min(center - radius)
        max_bound = max(center + radius)
        bounds.append([min_bound, max_bound])

        plt.vlines(e, bounds[-1][0], bounds[-1][1] )


    plt.xlabel('Epochs')
    plt.ylabel('Spectral bounds')
    plt.tight_layout()
    plt.show()
   
