###############################################################################
#
# Some code is adapted from https://github.com/KangchengHou/gntk
#
###############################################################################

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from gcn_rw_conv import gcn_rw_norm


def compute_margin_error(prob, label, margin):
    '''
        prob: np array, floar, N X C
        label: np array, floar, N
    '''

    yy = prob[np.arange(prob.shape[0]), label]
    prob_others = prob * 1.0
    prob_others[np.arange(prob.shape[0]), label] = np.amin(prob, axis=1)
    max_others = np.amax(prob_others, axis=1)
    return np.sum((yy <= max_others + margin).astype(np.float)) / prob.shape[0]


class Graph(object):
    def __init__(self, g, label, node_tags=None, node_feat=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_feat: a numpy float tensor, one-hot representation of the tag 
                        that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.max_neighbor = 0
        self.node_feat = 0 if node_feat is None else node_feat
        self.C_in = 0
        self.C_out = 0


def load_data(dataset, degree_as_tag):
    ''' dataset: name of dataset '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('../data/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_feat = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array(
                        [float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_feat.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_feat != []:
                node_feat = np.stack(node_feat)
                node_feature_flag = True
            else:
                node_feat = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(Graph(g, l, node_tags))

    max_degree = 0
    min_degree = 100000
    for g in g_list:
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
        g.min_neighbor = min(degree_list)
        if g.min_neighbor < min_degree:
            min_degree = g.min_neighbor

        g.label = label_dict[g.label]

        # edges = [list(pair) for pair in g.g.edges()]
        # edges.extend([[i, j] for j, i in edges])
        # deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())

    #Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    max_num_node, max_num_edge, max_norm, max_sym_adj_norm, max_rw_adj_norm = .0, .0, .0, .0, .0
    for g in g_list:
        g.node_feat = np.zeros([len(g.node_tags), len(tagset)])
        g.node_feat[range(len(g.node_tags)),
                    [tag2index[tag] for tag in g.node_tags]] = 1

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        edges.extend(
            [[i, i] for i in range(len(g.node_tags))])  # add self-loops
        g.C_in = np.zeros([len(g.node_tags), len(edges)])
        g.C_out = np.zeros([len(g.node_tags), len(edges)])
        e_ind, e_weights = get_laplacian(torch.LongTensor(edges).T, normalization='sym')
        g.Laplacian = to_dense_adj(e_ind, None, e_weights)[0]
        sym_adj_ind, sym_adj_weights = gcn_norm(torch.LongTensor(edges).T)
        sym_adj_norm = torch.norm(to_dense_adj(sym_adj_ind, None, sym_adj_weights)[0])
        if sym_adj_norm > max_sym_adj_norm:
            max_sym_adj_norm = sym_adj_norm
        rw_adj_ind, rw_adj_weights = gcn_rw_norm(torch.LongTensor(edges).T)
        rw_adj_norm = torch.norm(to_dense_adj(rw_adj_ind, None, rw_adj_weights)[0])
        if rw_adj_norm > max_rw_adj_norm:
            max_rw_adj_norm = rw_adj_norm

        norm = np.amax(np.sqrt((g.node_feat * g.node_feat).sum(axis=1)))
        num_node = len(g.node_tags)
        num_edge = len(edges)

        if num_node > max_num_node:
            max_num_node = num_node

        if num_edge > max_num_edge:
            max_num_edge = num_edge

        if norm > max_norm:
            max_norm = norm

        edges = sorted(edges)
        for ee, edge in enumerate(edges):
            g.C_in[edge[0], ee] = 1
            g.C_out[edge[1], ee] = 1

    G_sym = min(np.sqrt((max_degree + 1)/(min_degree + 1)), max_sym_adj_norm)
    G_rw = min(np.sqrt((max_degree + 1)/(min_degree + 1)), max_rw_adj_norm)

    print('Max node feature 2-norm: {}'.format(max_norm))
    print('Max node degree: {}'.format(max_degree))
    print('Min node degree: {}'.format(min_degree))
    print('Max # nodes: {}'.format(max_num_node))
    print('Max # edges: {}'.format(max_num_edge))
    print('# classes: {}'.format(len(label_dict)))
    print('# maximum node tag: {}'.format(len(tagset)))
    print("# data: {}".format(len(g_list)))
    print('G sym: {}'.format(G_sym))
    print('G rw: {}'.format(G_rw))

    return g_list, len(label_dict), max_norm, G_sym, G_rw, max_degree


def gen_synthetic_graphs(graph_type='ER',
                         num_graph=100,
                         num_node=100,
                         edge_prob=0.1,
                         node_feat_dim=16,
                         num_class=2,
                         max_feat_norm=1.0,
                         sizes=None,
                         probs=None,
                         seed=1234):
    g_list = []
    npr = np.random.RandomState(seed)
    if graph_type == 'ER':        
        for ii in range(num_graph):
            g = nx.erdos_renyi_graph(num_node, edge_prob, seed=seed + ii)
            rand_label = npr.randint(0, num_class)
            rand_feat = npr.randn(num_node, node_feat_dim)
            rand_feat_norm = np.sqrt((rand_feat * rand_feat).sum(
                axis=1, keepdims=True))
            rand_feat = (rand_feat / rand_feat_norm) * max_feat_norm
            g_list += [Graph(g, rand_label, node_feat=rand_feat)]
    elif graph_type == 'SBM':
        num_node = sum(sizes)
        for ii in range(num_graph):
            g = nx.stochastic_block_model(sizes, probs, seed=seed + ii)
            rand_label = npr.randint(0, num_class)
            rand_feat = npr.randn(num_node, node_feat_dim)
            rand_feat_norm = np.sqrt((rand_feat * rand_feat).sum(
                axis=1, keepdims=True))
            rand_feat = (rand_feat / rand_feat_norm) * max_feat_norm
            g_list += [Graph(g, rand_label, node_feat=rand_feat)]
    else:
        raise ValueError('Unsupported graph type!')

    max_num_node, max_num_edge, max_norm, max_degree, min_degree, max_sym_adj_norm, max_rw_adj_norm = .0, .0, .0, .0, 100000, .0, .0
    for g in g_list:
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        edges.extend([[i, i] for i in range(len(g.g.nodes))])  # add self-loops
        g.C_in = np.zeros([len(g.g.nodes), len(edges)])
        g.C_out = np.zeros([len(g.g.nodes), len(edges)])

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

        g.min_neighbor = min(degree_list)

        if g.min_neighbor < min_degree:
            min_degree = g.min_neighbor

        norm = np.amax(np.sqrt((g.node_feat * g.node_feat).sum(axis=1)))
        num_node = len(g.g.nodes)
        num_edge = len(edges)

        if num_node > max_num_node:
            max_num_node = num_node

        if num_edge > max_num_edge:
            max_num_edge = num_edge

        if norm > max_norm:
            max_norm = norm

        edges = sorted(edges)
        e_ind, e_weights = get_laplacian(torch.LongTensor(edges).T, normalization='sym')
        g.Laplacian = to_dense_adj(e_ind, None, e_weights)[0]
        sym_adj_ind, sym_adj_weights = gcn_norm(torch.LongTensor(edges).T)
        sym_adj_norm = torch.norm(to_dense_adj(sym_adj_ind, None, sym_adj_weights)[0])
        rw_adj_ind, rw_adj_weights = gcn_rw_norm(torch.LongTensor(edges).T)
        rw_adj_norm = torch.norm(to_dense_adj(rw_adj_ind, None, rw_adj_weights)[0])
        if rw_adj_norm > max_rw_adj_norm:
            max_rw_adj_norm = rw_adj_norm
        if sym_adj_norm > max_sym_adj_norm:
            max_sym_adj_norm = sym_adj_norm
        for ee, edge in enumerate(edges):
            g.C_in[edge[0], ee] = 1
            g.C_out[edge[1], ee] = 1

    G_sym = min(np.sqrt((max_degree + 1)/(min_degree + 1)), max_sym_adj_norm)
    G_rw = min(np.sqrt((max_degree + 1)/(min_degree + 1)), max_rw_adj_norm)
    print('Max node feature 2-norm: {}'.format(max_norm))
    print('Max node degree: {}'.format(max_degree))
    print('Min node degree: {}'.format(min_degree))
    print('Max # nodes: {}'.format(max_num_node))
    print('Max # edges: {}'.format(max_num_edge))
    print('G sym: {}'.format(G_sym))
    print('G rw: {}'.format(G_rw))

    return g_list, G_sym, G_rw
