import os
import json
import torch
import torch.nn as nn
import math

from utils import DocumentationConstants, get_model_hyperparameters, collate
from data_preprocess import get_datasets, get_dataloaders

from typing import Dict, List, Tuple
from data_preprocess import MoleculeDataset

def init_weights_linear(weight, bias):
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super(ResidualBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, input_dim))
            init_weights_linear(layers[-1].weight, layers[-1].bias)
            layers.append(nn.ReLU())
        layers.pop()  # delete in the last layer relu
        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):
        x_in = x
        x = self.linear_layers(x)
        out = x + x_in
        return out

class FFN(nn.Module):
    def __init__(self, structure: List[Tuple]): # ('R',2048), ('L',1024,2048)
        super(FFN, self).__init__()
        layers = []
        for structure_tuple in structure:
            if structure_tuple[0] == 'L':
                layers.append(nn.Linear(structure_tuple[1], structure_tuple[2]))
                init_weights_linear(layers[-1].weight, layers[-1].bias)
            elif structure_tuple[0] == 'R':
                layers.append(ResidualBlock(structure_tuple[1], 1))
            else:
                raise ValueError('Unknown layer type')
            layers.append(nn.ReLU())
        layers.pop()  # delete in the last layer relu
        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_layers(x)


 #hidden_dim, number_of_graph_convs, use_std = False
class NMRNet(nn.Module):
    def __init__(self, node_input_dim: int, edge_input_dim: int, net_parameters: Dict):
        super(NMRNet, self).__init__()
        self.number_of_graph_convs = net_parameters.get('graph_convs')

        self.relu = nn.ReLU()
        self.embedding_h = nn.Linear(node_input_dim, net_parameters.get('conv_dim'))
        self.embedding_e = nn.Linear(edge_input_dim, net_parameters.get('conv_dim'))

        # First convolution block is a special case
        self.graph_convs = nn.ModuleList(
            [GatedGCN_layer(net_parameters.get('conv_dim'), net_parameters.get('conv_dim'), res=(i % 2 == 0)) for i in
             range(self.number_of_graph_convs - 1)])
        self.MuBlock = FFN(net_parameters.get('mu_structure'))

        self.use_std = net_parameters.get('use_std')
        if self.use_std:
            self.SigmaBlock = FFN(net_parameters.get('std_structure'))

    def forward(self, g, X, E, snorm_n, snorm_e):
        # input embedding
        H = self.relu(self.embedding_h(X)) # how does backpropagation work here?
        E = self.relu(self.embedding_e(E)) # how does backpropagation work here?
        # Pass the molecule through each graph convolution block
        i = 0
        H_conv = H
        for conv in self.graph_convs:
            H_conv, E_emb = conv(g, H_conv, E, snorm_n, snorm_e)

        # Calculate the mean and standard deviation using the MuBlock and SigmaBlock
        mu = self.MuBlock(H_conv)
        if self.use_std:
            sigma = self.SigmaBlock(H)
        else:
            sigma = 0

        return mu, sigma


class GatedGCN_layer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, res: bool=True):
        super().__init__()
        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        self.epsilon = nn.Parameter(torch.Tensor([1e-5]))
        self.res = res
        self.init_weights([self.A, self.B, self.C, self.D, self.E])

    def init_weights(self, layers):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                init_weights_linear(layer.weight, layer.bias)

    def message_func(self, edges):
        Bx_j = edges.src['BX']
        # e_j = Ce_j + Dxj + Ex
        e_j = edges.data['CE'] + edges.src['DX'] + edges.dst['EX']
        edges.data['E'] = e_j
        return {'Bx_j': Bx_j, 'e_j': e_j}

    def reduce_func(self, nodes):
        Ax = nodes.data['AX']
        Bx_j = nodes.mailbox['Bx_j']
        e_j = nodes.mailbox['e_j']
        # sigma_j = σ(e_j)
        σ_j = torch.sigmoid(e_j)
        # h = Ax + Σ_j η_j * Bxj
        h = Ax + torch.sum(σ_j * Bx_j, dim=1) / (torch.sum(σ_j, dim=1) + self.epsilon)
        return {'H': h}

    def forward(self, g, X, E_X, snorm_n, snorm_e):
        g.ndata['H'] = X
        g.ndata['AX'] = self.A(X)
        g.ndata['BX'] = self.B(X)
        g.ndata['DX'] = self.D(X)
        g.ndata['EX'] = self.E(X)
        g.edata['E'] = E_X
        g.edata['CE'] = self.C(E_X)

        g.update_all(self.message_func, self.reduce_func)

        H = g.ndata['H']  # result of graph convolution
        E = g.edata['E']  # result of graph convolution

        H *= snorm_n  # normalize activation w.r.t. graph node size
        E *= snorm_e  # normalize activation w.r.t. graph edge size
        if self.res:
            H = self.bn_node_h(H)  # batch normalization
            E = self.bn_node_e(E)  # batch normalization

        H = torch.relu(H)  # non-linear activation
        E = torch.relu(E)  # non-linear activation

        if self.res:
            H = X + H  # residual connection
            E = E_X + E  # residual connection

        return H, E


def log_normal_nolog(y, mu, std):  # the loss functions
    element_wise = -(y - mu) ** 2 / (2 * std ** 2) - std
    return element_wise


class NormUncertainLoss(nn.Module):
    """
    Masked uncertainty loss
    """

    def __init__(self,
                 std_regularize=0.0,
                 ):
        super(NormUncertainLoss, self).__init__()
        self.std_regularize = std_regularize

    def __call__(self, mu, y, std):
        std = std + self.std_regularize
        return -log_normal_nolog(y,
                                 mu,
                                 std).mean()

def get_GGCNN_model(net_parameters: Dict) -> Tuple[nn.Module, List, List]:
    nmr_type = net_parameters.get('nmr_type')
    device = torch.device(net_parameters['device'])
    train_dataset, test_dataset = get_datasets(nmr_type, net_parameters.get('max_atoms'), net_parameters.get('max_bonds'),
                       net_parameters.get('features_set'), extend_label=net_parameters.get('extend_label'))
    node_input_shape = train_dataset[0][0].ndata['features'].shape[1]
    edge_input_shape = train_dataset[0][0].edata['features'].shape[1]
    model = NMRNet(node_input_shape, edge_input_shape, net_parameters).to(device)
    return model, train_dataset, test_dataset

class ModelLoader():
    def __init__(self, log_dir: str = None, net_parameters: Dict=None) -> None:
        if log_dir is None:
            log_dir = os.getcwd()
        self.log_dir = log_dir
        self.set_net_parameters(net_parameters)

    def set_net_parameters(self, net_parameters=None):
        if net_parameters:
            self.net_parameters = net_parameters
        elif self.log_dir:
            net_parameters_file_path = os.path.join(self.log_dir, DocumentationConstants.INFO_JSON_PATH.value)
            self.net_parameters = json.load(open(net_parameters_file_path, 'r'))
        else:
            net_parameters = get_model_hyperparameters()
        self.net_parameters['device'] = 'cuda' if torch.cuda.is_available() else 'cpu' # in case of loading and running on different machines
        nmr_type = self.net_parameters.get('nmr_type') # to fit old versions of the code
        if nmr_type not in ('H_NMR', 'C_NMR'):
            self.net_parameters['nmr_type'] = 'H_NMR'

    def load_model(self) -> Tuple[nn.Module, List, List, MoleculeDataset]:
        self.model, self.train_dataset, self.test_dataset = get_GGCNN_model(self.net_parameters)
        self.features_meta_data = self.train_dataset.metadata
        self.train_dataloader, self.test_dataloader = get_dataloaders(self.train_dataset, self.test_dataset,
                                                                      batch_size=self.net_parameters.get('batch_size'), shuffle=True, collate=collate)

        return self.model, self.train_dataloader, self.test_dataloader, self.train_dataset


if __name__ == '__main__':
    pass
