import torch
import numpy as np

def get_splits(dataset_name, graphs, seed, use_synthetic_graphs=True, train_ratio=0.9):
    npr = np.random.RandomState(seed)
    num_graphs = len(graphs)
    
    if use_synthetic_graphs:
        rand_idx = npr.permutation(num_graphs)
        num_train = int(num_graphs * train_ratio)            
        train_idx = rand_idx[:num_train].tolist()
        test_idx = rand_idx[num_train:].tolist()
    else:
        with open('../data/{}/10fold_idx/train_idx-1.txt'.format(dataset_name),
                'r') as ff:
            train_idx = [int(xx) for xx in ff.readlines()]

        with open('../data/{}/10fold_idx/test_idx-1.txt'.format(dataset_name),
                'r') as ff:
            test_idx = [int(xx) for xx in ff.readlines()]

    return train_idx, test_idx


class GraphData(object):
    def __init__(self, dataset_name, graphs, split='train', use_synthetic_graphs=False, train_ratio=0.9):
        assert split == 'train' or split == 'test', "no such split"
        self.split = split        
        num_graphs = len(graphs)
        
        npr = np.random.RandomState(1234)

        ### split
        if use_synthetic_graphs:
            rand_idx = npr.permutation(num_graphs)
            num_train = int(num_graphs * train_ratio)            
            train_idx = rand_idx[:num_train].tolist()
            test_idx = rand_idx[num_train:].tolist()
        else:
            with open('../data/{}/10fold_idx/train_idx-1.txt'.format(dataset_name),
                      'r') as ff:
                train_idx = [int(xx) for xx in ff.readlines()]

            with open('../data/{}/10fold_idx/test_idx-1.txt'.format(dataset_name),
                      'r') as ff:
                test_idx = [int(xx) for xx in ff.readlines()]

        self.num_train = len(train_idx)
        self.num_test = len(test_idx)
        self.num_graphs = len(graphs)
        self.train_graphs = [graphs[ii] for ii in train_idx]
        self.test_graphs = [graphs[ii] for ii in test_idx]

    def __getitem__(self, index):
        if self.split == 'train':
            return self.train_graphs[index]
        else:
            return self.test_graphs[index]

    def __len__(self):
        if self.split == 'train':
            return self.num_train
        else:
            return self.num_test

    def collate_fn(self, batch):
        assert isinstance(batch, list)

        data = {}
        batch_size = len(batch)
        node_size = [bb.node_feat.shape[0] for bb in batch]
        edge_size = [bb.C_in.shape[1] for bb in batch]
        N = max(node_size)
        E = max(edge_size)
        pad_node_size = [N - nn for nn in node_size]
        pad_edge_size = [E - nn for nn in edge_size]

        # pad feature: shape (B, N, D)
        data['node_feat'] = torch.stack([
            torch.from_numpy(
                np.pad(
                    bb.node_feat, ((0, pad_node_size[ii]), (0, 0)),
                    'constant',
                    constant_values=0.0)) for ii, bb in enumerate(batch)
        ]).float()

        # pad Laplacian: shape (B, N, N)
        data['Laplacian'] = torch.stack([
            torch.from_numpy(
                np.pad(
                    bb.Laplacian, ((0, pad_node_size[ii]), (0, pad_node_size[ii])),
                    'constant',
                    constant_values=0.0)) for ii, bb in enumerate(batch)
        ]).float()

        # pad C_in: shape (B, N, E)
        data['C_in'] = torch.stack([
            torch.from_numpy(
                np.pad(
                    bb.C_in, ((0, pad_node_size[ii]), (0, pad_edge_size[ii])),
                    'constant',
                    constant_values=0.0)) for ii, bb in enumerate(batch)
        ]).float()

        # pad C_out: shape (B, N, E)
        data['C_out'] = torch.stack([
            torch.from_numpy(
                np.pad(
                    bb.C_out, ((0, pad_node_size[ii]), (0, pad_edge_size[ii])),
                    'constant',
                    constant_values=0.0)) for ii, bb in enumerate(batch)
        ]).float()

        # binary mask: shape (B, N)
        data['mask'] = torch.stack([
            torch.from_numpy(
                np.pad(
                    np.ones(node_size[ii]), (0, pad_node_size[ii]),
                    'constant',
                    constant_values=0.0)) for ii, bb in enumerate(batch)
        ]).byte()

        # label: shape (B)
        data['label'] = torch.stack(
            [torch.from_numpy(np.array(bb.label)) for bb in batch])

        return data
