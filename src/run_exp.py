import os
import pickle
import argparse
import logging

import numpy as np
import torch
import torch.optim as optim
import torch_geometric
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
import pandas as pd

from models import MPGNN, GCN, GCN_RW
from dataset import GraphData, get_splits
from utils import load_data, gen_synthetic_graphs

logging.basicConfig(
    format=
    "%(levelname)-5s | %(asctime)s | File %(filename)-20s | Line %(lineno)-5d | %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG)

### Parse Arguments
parser = argparse.ArgumentParser(description='GNN Experiments')
# several folders, each folder one kernel
parser.add_argument(
    '--dataset',
    type=str,
    default="ER-5",
    help='name of dataset')
parser.add_argument(
    '--hidden_dim', type=int, default=32, help='hidden dimension')
parser.add_argument(
    '--train_ratio', type=float, default=0.1, help='training ratio')
parser.add_argument(
    '--max_epoch', type=int, default=200, help='maximum number of epochs')
parser.add_argument(
    '--batch_size', type=int, default=128, help='batch size')
parser.add_argument(
    '--alpha', type=float, default=100, help='alpha in weight decay and in bounds')
parser.add_argument(
    '--delta', type=float, default=0.05, help='(1-delta) is the probability of the Rademacher bound')
parser.add_argument(
    '--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument(
    '--eval_bound_only',
    type=bool,
    default=False,
    help='evaluate trained model')
parser.add_argument(
    '--model_path', type=str, default='exp/', help='path of trained model')
parser.add_argument(
    '--with_constants',
    type=bool,
    default=True,
    help='compute bound with constants')

parser.add_argument(
    '--load_only',
    default=False,
    action='store_true', 
    help='load saved model without training')

parser.add_argument(
        '--model_type',
        type=str,
        default='GCN',
        help='type of model used')

parser.add_argument(
        '--pooling_method',
        type=str,
        default='mean',
        help='type of pooling used')

parser.add_argument('--seed', type=int, default=10, help='random seed')
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)

if args.dataset == 'PROTEINS':
    degree_as_tag = False  # bioinformatics

batch_size = args.batch_size
device = torch.device('cuda:{}'.format(args.gpu_id))
margin = 1


if args.dataset in ['PROTEINS']:
    ### Load real-world data
    graphs, num_class, max_feat_norm, G_sym, G_rw, max_node_degree = load_data(
        args.dataset, degree_as_tag)
    input_dim = graphs[0].node_feat.shape[1]
    if args.train_ratio == 0.9: # could use default splits
        use_synthetic_graphs = False
    else:
        use_synthetic_graphs = True
else:
    # synthetic data
    use_synthetic_graphs = True
    save_path = os.path.join(os.path.dirname(os.path.realpath(
            __file__)), '../data/'+args.dataset+'/seed'+str(args.seed)+'.pk')
    if os.path.exists(save_path):
        print('Loading existing data!')
        data = pickle.load(open(save_path, 'rb'))
        graphs = data['graphs']
        G_sym = data['G_sym']
        G_rw = data['G_rw']
        input_dim = data['input_dim']
        num_class = data['num_class']
        max_feat_norm = data['max_feat_norm']
        # then compute max_node_degree
        max_degree = 0
        for g in graphs:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)

            if g.max_neighbor > max_degree:
                max_degree = g.max_neighbor

            max_node_degree = max_degree
    else:
        print('Generating new data!')
        if args.dataset == 'SBM-1':
            num_graph = 200
            num_node = 100
            sizes = [40, 60]
            probs = [[0.25, 0.13], [0.13, 0.37]]
            node_feat_dim = 16
            num_class = 2
            max_feat_norm = 1.0
            graphs, G_sym, G_rw = gen_synthetic_graphs(
                graph_type='SBM',
                num_graph=num_graph,
                sizes=sizes,
                probs=probs,
                node_feat_dim=node_feat_dim,
                num_class=num_class,
                max_feat_norm=max_feat_norm,
                seed=args.seed)  # this seed controls dataset
            input_dim = node_feat_dim
            use_synthetic_graphs = True
        elif args.dataset == 'SBM-2':
            num_graph = 200
            num_node = 100
            sizes = [25, 25, 50]
            probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
            node_feat_dim = 16
            num_class = 2
            max_feat_norm = 1.0
            graphs, G_sym, G_rw = gen_synthetic_graphs(
                graph_type='SBM',
                num_graph=num_graph,
                sizes=sizes,
                probs=probs,
                node_feat_dim=node_feat_dim,
                num_class=num_class,
                max_feat_norm=max_feat_norm,
                seed=args.seed)  # this seed controls dataset
            input_dim = node_feat_dim
            use_synthetic_graphs = True
        elif args.dataset == 'SBM-3':
            num_graph = 200
            num_node = 50
            sizes = [15, 15, 20]
            probs = [[0.5, 0.1, 0.2], [0.1, 0.4, 0.1], [0.2, 0.1, 0.4]]
            node_feat_dim = 16
            num_class = 2
            max_feat_norm = 1.0
            graphs, G_sym, G_rw = gen_synthetic_graphs(
                graph_type='SBM',
                num_graph=num_graph,
                sizes=sizes,
                probs=probs,
                node_feat_dim=node_feat_dim,
                num_class=num_class,
                max_feat_norm=max_feat_norm,
                seed=args.seed)  # this seed controls dataset
            input_dim = node_feat_dim
            use_synthetic_graphs = True
        elif args.dataset == 'ER-4':
            ### Load synthetic data
            num_graph = 200
            num_node = 100
            edge_prob = 0.7
            node_feat_dim = 16
            num_class = 2
            max_feat_norm = 1.0
            graphs, G_sym, G_rw = gen_synthetic_graphs(
                graph_type='ER',
                num_graph=num_graph,
                num_node=num_node,
                edge_prob=edge_prob,
                node_feat_dim=node_feat_dim,
                num_class=num_class,
                max_feat_norm=max_feat_norm,
                seed=args.seed)  # this seed controls dataset
            input_dim = node_feat_dim
            use_synthetic_graphs = True
        elif args.dataset == 'ER-5':
            ### Load synthetic data
            num_graph = 200
            num_node = 20
            edge_prob = 0.5
            node_feat_dim = 16
            num_class = 2
            max_feat_norm = 1.0
            graphs, G_sym, G_rw = gen_synthetic_graphs(
                graph_type='ER',
                num_graph=num_graph,
                num_node=num_node,
                edge_prob=edge_prob,
                node_feat_dim=node_feat_dim,
                num_class=num_class,
                max_feat_norm=max_feat_norm,
                seed=args.seed)  # this seed controls dataset
            input_dim = node_feat_dim
            use_synthetic_graphs = True

        data = Data(graphs=graphs, G_sym=G_sym, G_rw=G_rw, input_dim=input_dim, num_class=num_class, max_feat_norm=max_feat_norm)
        if os.path.isdir(os.path.dirname(save_path)) == False:
            try:
                os.makedirs(os.path.dirname(save_path))
            except FileExistsError:
                print('Folder exists for {}!'.format(os.path.dirname(save_path)))
        pickle.dump(data, open(save_path, 'wb'))


### Create Model
if args.model_type == 'MPGNN':

    train_dataset = GraphData(
        args.dataset,
        graphs,
        split='train',
        use_synthetic_graphs=use_synthetic_graphs,
        train_ratio=args.train_ratio)
    test_dataset = GraphData(
        args.dataset,
        graphs,
        split='test',
        use_synthetic_graphs=use_synthetic_graphs,
        train_ratio=args.train_ratio)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=test_dataset.collate_fn)

    N_max = 0
    for data in train_loader:
        current_N = data['node_feat'].shape[0]
        if current_N > N_max:
            N_max = current_N
    model = MPGNN(input_dim,
                  args.hidden_dim, num_class, max_feat_norm,
                  len(train_dataset), G_sym, margin, args.pooling_method, N_max).to(device)
elif args.model_type[:3] == 'GCN':
    train_idx, test_idx = get_splits(args.dataset, graphs, args.seed, use_synthetic_graphs, args.train_ratio)
    train_data = []
    test_data = []
    for ii in train_idx:
        g = from_networkx(graphs[ii].g)
        g.x = torch.Tensor(graphs[ii].node_feat).to(device)
        g.label = 2 * (graphs[ii].label - 0.5)
        train_data.append(g)

    for ii in test_idx:
        g = from_networkx(graphs[ii].g)
        g.x = torch.Tensor(graphs[ii].node_feat).to(device)
        g.label = 2 * (graphs[ii].label - 0.5)
        test_data.append(g)

    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch_geometric.loader.DataLoader(test_data, batch_size=batch_size)
    
    N_max = 0
    for data in train_loader:
        current_N = data.x.shape[0]
        if current_N > N_max:
            N_max = current_N
    if args.model_type == 'GCN':
        model = GCN(input_dim,
                    args.hidden_dim, num_class, max_feat_norm,
                    len(train_data), G_sym, margin, device, args.pooling_method, N_max).to(device)
    elif args.model_type == 'GCN_RW':
        model = GCN_RW(input_dim,
                args.hidden_dim, num_class, max_feat_norm,
                len(train_data), G_rw, margin, device, args.pooling_method, N_max).to(device)

### Create Optimizer
optimizer = optim.SGD(model.parameters(), lr=5.0e-3, momentum=0.9, weight_decay=1/(args.alpha*args.hidden_dim))

### Training Loop
stats = {
    'excess_risk': [],
    'generalization_error_bound': [], # from functional derivative
    'generalization_error_bound_Rademacher': []
}

exp_folder = '../exp/{}_{}'.format(args.model_type, args.pooling_method)
if not os.path.isdir(exp_folder):
    os.makedirs(exp_folder)

excess_risk_list = []
generalization_bound_list = []
generalization_bound_Rademacher_list = []

if args.load_only:
    model_snapshot = torch.load(
            os.path.join(
                exp_folder, "{}_model_snapshot_{:07d}_seed_{}_hidden{}_100train_ratio{}_alpha{}.pth".format(
                    args.dataset, args.max_epoch, args.seed, args.hidden_dim, int(100*args.train_ratio), args.alpha)))
    model.load_state_dict(model_snapshot['model'])
    if args.model_type[:3] == 'GCN':
        train_logit = []
        for data in train_loader:
            logit = model(data.x.to(device),
                            data.edge_index.to(device),
                            data.batch.to(device), 
                            data.label.to(device))

            train_logit += [logit.cpu().data.numpy()]

        train_error = np.array(train_logit).mean()

        test_logit = []
        for data in test_loader:
            logit = model(data.x.to(device),
                            data.edge_index.to(device),
                            data.batch.to(device), 
                            data.label.to(device))

            test_logit += [logit.cpu().data.numpy()]
    elif args.model_type == 'MPGNN':
        train_logit = []
        for data in train_loader:
            logit = model(
                data['node_feat'].to(device),
                data['Laplacian'].to(device),
                label=data['label'].to(device))

            train_logit += [logit.cpu().data.numpy()]

        train_error = np.array(train_logit).mean()

        test_logit = []
        for data in test_loader:
            logit = model(
                data['node_feat'].to(device),
                data['Laplacian'].to(device),
                label=data['label'].to(device))
                
            test_logit += [logit.cpu().data.numpy()]
        
    test_error = np.array(test_logit).mean()
    excess_risk = test_error - train_error
    stats['excess_risk'] = excess_risk
    if args.pooling_method == 'mean':
        error_bound = model.mean_pooling_generalization_bound(args.alpha).data.cpu().numpy()[0]
        error_bound_Rademacher = model.Rademacher_mean_pooling_generalization_bound(args.alpha, max_node_degree, args.delta).data.cpu().numpy()[0]
    elif args.pooling_method == 'max':
        error_bound = 0 # no bound computed for now
        error_bound_Rademacher = 0
    elif args.pooling_method == 'sum':
        error_bound = model.sum_pooling_generalization_bound(args.alpha).data.cpu().numpy()[0]
        error_bound_Rademacher = model.Rademacher_sum_pooling_generalization_bound(args.alpha, max_node_degree, args.delta).data.cpu().numpy()[0]
    stats['generalization_error_bound'] = error_bound
    stats['generalization_error_bound_Rademacher'] = error_bound_Rademacher
    logging.info('Load only: Excess Risk: {:.6f}, Generalization Error Bound: {:.6f}, Generalization Error Bound (Rademacher): {:.6f}.'.format(
        excess_risk, error_bound, error_bound_Rademacher))
    excess_risk_list.append(excess_risk)
    generalization_bound_list.append(error_bound)
    generalization_bound_Rademacher_list.append(error_bound_Rademacher)
else:
    for epoch in range(args.max_epoch):  # loop over the dataset multiple times
        if (epoch+1) % 10 == 0: # excess risk and bounds
            if args.model_type[:3] == 'GCN':
                train_logit = []
                for data in train_loader:
                    logit = model(data.x.to(device),
                                    data.edge_index.to(device),
                                    data.batch.to(device), 
                                    data.label.to(device))

                    train_logit += [logit.cpu().data.numpy()]

                train_error = np.array(train_logit).mean()

                test_logit = []
                for data in test_loader:
                    logit = model(data.x.to(device),
                                    data.edge_index.to(device),
                                    data.batch.to(device), 
                                    data.label.to(device))

                    test_logit += [logit.cpu().data.numpy()]
            elif args.model_type == 'MPGNN':
                train_logit = []
                for data in train_loader:
                    logit = model(
                        data['node_feat'].to(device),
                        data['Laplacian'].to(device),
                        label=data['label'].to(device))

                    train_logit += [logit.cpu().data.numpy()]

                train_error = np.array(train_logit).mean()

                test_logit = []
                for data in test_loader:
                    logit = model(
                        data['node_feat'].to(device),
                        data['Laplacian'].to(device),
                        label=data['label'].to(device))
                        
                    test_logit += [logit.cpu().data.numpy()]
                
            test_error = np.array(test_logit).mean()
            excess_risk = test_error - train_error
            stats['excess_risk'] = excess_risk
            if args.pooling_method == 'mean':
                error_bound = model.mean_pooling_generalization_bound(args.alpha).data.cpu().numpy()[0]
                error_bound_Rademacher = model.Rademacher_mean_pooling_generalization_bound(args.alpha, max_node_degree, args.delta).data.cpu().numpy()[0]
            elif args.pooling_method == 'max':
                error_bound = 0 # no bound computed for now
                error_bound_Rademacher = 0 # no bound computed for now
            elif args.pooling_method == 'sum':
                error_bound = model.sum_pooling_generalization_bound(args.alpha).data.cpu().numpy()[0]
                error_bound_Rademacher = model.Rademacher_sum_pooling_generalization_bound(args.alpha, max_node_degree, args.delta).data.cpu().numpy()[0]
            stats['generalization_error_bound'] = error_bound
            stats['generalization_error_bound_Rademacher'] = error_bound_Rademacher
            logging.info('Epoch: {:07d}, Excess Risk: {:.6f}, Generalization Error Bound: {:.6f}, Generalization Error Bound (Rademacher): {:.6f}.'.format(
                epoch+1, excess_risk, error_bound, error_bound_Rademacher))
            excess_risk_list.append(excess_risk)
            generalization_bound_list.append(error_bound)
            generalization_bound_Rademacher_list.append(error_bound_Rademacher)
        # training
        running_loss = .0
        for data in train_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            if args.model_type == 'MPGNN':
                loss = model(
                    data['node_feat'].to(device),
                    data['Laplacian'].to(device),
                    label=data['label'].to(device))
            elif args.model_type[:3] == 'GCN':
                loss = model(data.x.to(device),
                                data.edge_index.to(device),
                                data.batch.to(device), 
                                data.label.to(device))

            loss.backward()
            optimizer.step()
            running_loss = 0.8 * loss.item() + 0.2 * running_loss

        if (epoch + 1) % 10 == 0:
            model_snapshot = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": epoch
            }
            torch.save(
                model_snapshot,
                os.path.join(
                    exp_folder, "{}_model_snapshot_{:07d}_seed_{}_hidden{}_100train_ratio{}_alpha{}.pth".format(
                        args.dataset, epoch + 1, args.seed, args.hidden_dim, int(100*args.train_ratio), args.alpha)))

        logging.info('Epoch: {:07d}, Loss: {:.3f}'.format(
            epoch + 1, running_loss))

save_name_str = '{}_{}_seed_{}_{}_{}_hidden{}_100train_ratio{}_alpha{}'.format(
                args.dataset, 'w_consts' if args.with_constants else
                'wo_consts', args.seed, args.model_type, args.pooling_method, args.hidden_dim, int(100*args.train_ratio), 
                args.alpha)
with open(
        os.path.join(
            exp_folder, save_name_str + '.p'), "wb") as ff:
    pickle.dump(stats, ff)
file_name = '../result_arrays/' + save_name_str
if not args.load_only:
    np.save(file_name+'_generalization_bound.npy', np.array(generalization_bound_list))
    np.save(file_name+'_generalization_bound_Rademacher.npy', np.array(generalization_bound_Rademacher_list))
    np.save(file_name+'_excess_risk.npy', np.array(excess_risk_list))

    logging.info('Finished Training')
else:
    np.save(file_name+'_generalization_bound.npy', np.array(generalization_bound_list))
    np.save(file_name+'_generalization_bound_Rademacher.npy', np.array(generalization_bound_Rademacher_list))
    np.save(file_name+'_excess_risk.npy', np.array(excess_risk_list))
