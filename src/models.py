import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from gcn_rw_conv import GCN_RWConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, 
            max_feat_norm, num_sample, G_value, margin, device, pooling_method, N_max):

        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.max_feat_norm = max_feat_norm
        self.G_value = G_value
        self.num_sample = num_sample
        self.margin = margin
        self.device = device

        self.conv_layer = GCNConv(self.input_dim, self.hidden_dim, bias=False).to(self.device)

        self.wl = GCNConv(self.hidden_dim, 1, bias=False).to(self.device) # change to 1D output
        if pooling_method == 'mean':
            self.pooling = global_mean_pool
        elif pooling_method == 'max':
            self.pooling = global_max_pool
        elif pooling_method == 'sum':
            self.pooling = global_add_pool
        self.N_max = N_max

    def loss_func(self, pooled, label):
        loss_val = torch.log(1 + torch.exp(-pooled*label)).mean()
        return loss_val
    
    def mean_pooling_generalization_bound(self, alpha):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_phi = w_m * self.max_feat_norm * w_1
        M_phi = torch.min(M_phi, w_m)
        bound = alpha * M_phi * M_phi * self.G_value * self.G_value /self.num_sample
        return bound

    def sum_pooling_generalization_bound(self, alpha):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_phi = w_m * self.max_feat_norm * w_1 * self.N_max
        M_phi = torch.min(M_phi, w_m * self.N_max)
        bound = alpha * M_phi * M_phi * self.G_value * self.G_value /self.num_sample
        return bound

    def Rademacher_mean_pooling_generalization_bound(self, alpha, max_node_degree, delta=0.05):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_l = torch.log(1 + torch.exp(w_m))
        M_phi = w_m * self.max_feat_norm * np.sqrt(max_node_degree+1) * w_1
        M_phi = torch.min(M_phi, w_m)
        bound = 4 * M_phi * torch.sqrt(M_phi*alpha/self.num_sample) + 3 * M_l * np.sqrt(np.log(2/delta)/2/self.num_sample)
        return bound

    def Rademacher_sum_pooling_generalization_bound(self, alpha, max_node_degree, delta=0.05):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_l = torch.log(1 + torch.exp(w_m))
        M_phi = w_m * self.max_feat_norm * np.sqrt(max_node_degree+1) * w_1 * self.N_max
        M_phi = torch.min(M_phi, w_m * self.N_max)
        bound = 4 * M_phi * torch.sqrt(M_phi*alpha/self.num_sample) + 3 * M_l * np.sqrt(np.log(2/delta)/2/self.num_sample)
        return bound

    def forward(self, x, edge_index, batch, label=None):
        x = self.conv_layer(x, edge_index)
        x = x.tanh()

        x = self.wl(x, edge_index)/self.hidden_dim
        pooled = self.pooling(x, batch).flatten()

        return self.loss_func(pooled, label)

class GCN_RW(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, 
            max_feat_norm, num_sample, G_value, margin, device, pooling_method, N_max):

        super(GCN_RW, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.max_feat_norm = max_feat_norm
        self.G_value = G_value
        self.num_sample = num_sample
        self.margin = margin
        self.device = device

        self.conv_layer = GCNConv(self.input_dim, self.hidden_dim, bias=False).to(self.device)

        self.wl = GCN_RWConv(self.hidden_dim, 1, bias=False).to(self.device) # change to 1D output
        if pooling_method == 'mean':
            self.pooling = global_mean_pool
        elif pooling_method == 'max':
            self.pooling = global_max_pool
        elif pooling_method == 'sum':
            self.pooling = global_add_pool
        self.N_max = N_max

    def loss_func(self, pooled, label):
        loss_val = torch.log(1 + torch.exp(-pooled*label)).mean()
        return loss_val
    
    def mean_pooling_generalization_bound(self, alpha):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_phi = w_m * self.max_feat_norm * w_1
        M_phi = torch.min(M_phi, w_m)
        bound = alpha * M_phi * M_phi * self.G_value * self.G_value /self.num_sample
        return bound

    def sum_pooling_generalization_bound(self, alpha):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_phi = w_m * self.max_feat_norm * w_1 * self.N_max
        M_phi = torch.min(M_phi, w_m * self.N_max)
        bound = alpha * M_phi * M_phi * self.G_value * self.G_value /self.num_sample
        return bound

    def Rademacher_mean_pooling_generalization_bound(self, alpha, max_node_degree, delta=0.05):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_l = torch.log(1 + torch.exp(w_m))
        M_phi = w_m * self.max_feat_norm * np.sqrt(max_node_degree+1) * w_1
        M_phi = torch.min(M_phi, w_m)
        bound = 4 * M_phi * torch.sqrt(M_phi*alpha/self.num_sample) + 3 * M_l * np.sqrt(np.log(2/delta)/2/self.num_sample)
        return bound

    def Rademacher_sum_pooling_generalization_bound(self, alpha, max_node_degree, delta=0.05):
        w_m = self.wl.lin.weight.max(dim=1)[0]
        w_1 = self.conv_layer.lin.weight.norm(dim=1).max()
        M_l = torch.log(1 + torch.exp(w_m))
        M_phi = w_m * self.max_feat_norm * np.sqrt(max_node_degree+1) * w_1 * self.N_max
        M_phi = torch.min(M_phi, w_m * self.N_max)
        bound = 4 * M_phi * torch.sqrt(M_phi*alpha/self.num_sample) + 3 * M_l * np.sqrt(np.log(2/delta)/2/self.num_sample)
        return bound

    def forward(self, x, edge_index, batch, label=None):
        x = self.conv_layer(x, edge_index)
        x = x.tanh()

        x = self.wl(x, edge_index)/self.hidden_dim
        pooled = self.pooling(x, batch).flatten()

        return self.loss_func(pooled, label)


class MPGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class,
                 max_feat_norm, num_sample, G_value, margin, pooling_method, N_max):
        super(MPGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = num_class
        self.max_feat_norm = max_feat_norm
        self.G_value = G_value
        self.num_sample = num_sample
        self.margin = margin

        self.W1 = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.W2 = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.b2 = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.Wl = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.bl = nn.Parameter(torch.Tensor(1))
        self.pooling_method = pooling_method
        self.N_max = N_max

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wl, a=math.sqrt(5))

        def bias_bound(weight):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            return 1 / math.sqrt(fan_in)

        bound1 = bias_bound(self.W1)
        nn.init.uniform_(self.b1, -bound1, bound1)

        bound2 = bias_bound(self.W2)
        nn.init.uniform_(self.b2, -bound2, bound2)

        boundl = bias_bound(self.Wl)
        nn.init.uniform_(self.bl, -boundl, boundl)

    def loss_func(self, pooled, label):
        loss_val = torch.log(1 + torch.exp(-pooled*label)).mean()
        return loss_val
    
    def mean_pooling_generalization_bound(self, alpha):
        w_m = self.Wl.max(dim=0)[0]
        w_3 = self.W2.norm(dim=0).max()
        w_1 = self.W1.norm(dim=0).max()
        M_phi = w_m * self.max_feat_norm * (w_3 + self.G_value * w_1)
        M_phi = torch.min(M_phi, w_m)
        bound = alpha * M_phi * M_phi/self.num_sample
        return bound
    
    def sum_pooling_generalization_bound(self, alpha):
        w_m = self.Wl.max(dim=0)[0]
        w_3 = self.W2.norm(dim=0).max()
        w_1 = self.W1.norm(dim=0).max()
        M_phi = w_m * self.max_feat_norm * (w_3 + self.G_value * w_1) * self.N_max
        M_phi= torch.min(M_phi, w_m * self.N_max)
        bound = alpha * M_phi * M_phi/self.num_sample
        return bound

    def Rademacher_mean_pooling_generalization_bound(self, alpha, max_node_degree, delta=0.05):
        w_m = self.Wl.max(dim=0)[0]
        w_3 = self.W2.norm(dim=0).max()
        w_1 = self.W1.norm(dim=0).max()
        M_l = torch.log(1 + torch.exp(w_m))
        M_phi = w_m * self.max_feat_norm * (w_3 + np.sqrt(max_node_degree+1) * w_1)
        M_phi = torch.min(M_phi, w_m)
        bound = 4 * M_phi * torch.sqrt(M_phi*alpha/self.num_sample) + 3 * M_l * np.sqrt(np.log(2/delta)/2/self.num_sample)
        return bound

    def Rademacher_sum_pooling_generalization_bound(self, alpha, max_node_degree, delta=0.05):
        w_m = self.Wl.max(dim=0)[0]
        w_3 = self.W2.norm(dim=0).max()
        w_1 = self.W1.norm(dim=0).max()
        M_l = torch.log(1 + torch.exp(w_m))
        M_phi = w_m * self.max_feat_norm * (w_3 + np.sqrt(max_node_degree+1) * w_1) * self.N_max
        M_phi = torch.min(M_phi, w_m * self.N_max)
        bound = 4 * M_phi * torch.sqrt(M_phi*alpha/self.num_sample) + 3 * M_l * np.sqrt(np.log(2/delta)/2/self.num_sample)
        return bound

    def forward(self, node_feat, Laplacian, label=None, mask=None):
        """
            shape parameters:
                batch size = B
                feature dim = D
                hidden dim = H
                max # nodes per batch = N
                max # edges per batch = E
                # classes = C

            Args:
                node_feat: float, shape B X N X D
                Laplacian: float, shape B X N X N          
                label: float, shape B X 1
                mask: float, shape B X N
        """
        # propagation
        # compute message, g = tanh
        msg = torch.tanh(node_feat)  # shape: B X N X H

        # aggregate message
        msg = Laplacian.bmm(msg)  # shape: B X N X H

        # update state, phi = ReLU, rho = tanh
        state = torch.tanh(
            node_feat.matmul(self.W1) +
            torch.tanh(msg).matmul(self.W2))

        # readout
        state = state.matmul(self.Wl) # + self.bl

        if self.pooling_method == 'mean':
            logit = (state.mean(dim=1)/self.hidden_dim).flatten() # shape: B
        elif self.pooling_method == 'max':
            logit = ((state.max(dim=1)[0])/self.hidden_dim).flatten()
        elif self.pooling_method == 'sum':
            logit = (state.sum(dim=1)/self.hidden_dim).flatten() # shape: B

        return self.loss_func(logit, label)