import os
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import dgl
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Dict, Union, Any
from rdkit.Chem.rdchem import Mol, Atom, Bond

from enum import Enum

class NativeColumnNames(Enum):
    X_DATA = 'rdmol'
    Y_DATA = 'value'

class Masks(Enum):
    MISSING_LABEL = np.inf # MASKING with inf - consider alternatives

class TENSORBOARD_SCALARS(Enum):
    TRAIN_LOSS = 'training loss'
    TEST_LOSS = 'test loss'
    TRAIN_AVG_ABS_ERROR = 'training avg abs error'
    TRAIN_MAX_ABS_ERROR = 'training max abs error'
    TEST_AVG_ABS_ERROR = 'test avg abs error'
    TEST_MAX_ABS_ERROR = 'test max abs error'

class DocumentationConstants(Enum):
    MAIN_DIR = "logs.{}/{}"
    INFO_DIR = 'info'
    INFO_JSON = 'training_info.json'
    INFO_TXT = 'training_info.txt'
    INFO_JSON_PATH = os.path.join(INFO_DIR, INFO_JSON)
    INFO_TXT_PATH = os.path.join(INFO_DIR, INFO_TXT)
    INFO_FEATURES_JSON = 'features_info.json'
    INFO_FEATURES_TXT = 'features_info.txt'
    INFO_FEATURES_PICKLE = 'features_info.pkl'
    INFO_FEATURES_JSON_PATH = os.path.join(INFO_DIR, INFO_FEATURES_JSON)
    INFO_FEATURES_TXT_PATH = os.path.join(INFO_DIR, INFO_FEATURES_TXT)
    INFO_FEATURES_PICKLE_PATH = os.path.join(INFO_DIR, INFO_FEATURES_PICKLE)
    CHECKPOINTS_DIR = 'checkpoints'
    CHECKPOINTS_PTH = 'checkpoint_epoch_{}.pth'
    CHECKPOINTS_PTH_PATH = os.path.join(CHECKPOINTS_DIR, CHECKPOINTS_PTH)
    TENSORBOARD_DIR = 'tensorboard'

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def get_model_hyperparameters() -> Dict:
    base_lr = 0.00001
    net_parameters = {'nmr_type': 'H_NMR', 'features_set': 1, 'use_std': False, 'max_atoms': 64, 'max_bonds': 50,
                      'batch_size': 64, 'graph_convs': 10, 'num_epochs': 200, 'extend_label': False, 'reduce_label': False, 'shuffle_labels': False,
                      'inital_lr': base_lr, 'checkpoint_every_k_epochs': 50, "MIN_LABEL": -5, "MAX_LABEL": 15,
                      'scheduler': {'use_scheduler': True, 'type': 'warm up', 'kwargs': {'start_factor': 1, 'end_factor': 0.05, 'total_iters':10}},
                    #   'scheduler': {'use_scheduler': True, 'type': 'cycle', 'kwargs': {'base_lr': base_lr, 'max_lr': 10*base_lr, 'step_size_up': 5, 'mode': 'triangular', 'cycle_momentum': False}},
                    #   'scheduler': {'use_scheduler': True, 'type': 'cosine anneal', 'kwargs': {'T_max': 10, 'eta_min': 0}},
                      'write_to_tensorboard': True, 'save_checkpoints': True, 'optimizer': 'adam',
                      'mu_structure': [('R',2048), ('R',2048), ('R',2048), ('L',2048,1024), ('L',1024,512), ('L',512,256), ('L',256,128), ('L',128,1)],
                      'std_structure' : [('L',2048,2048), ('L',1024,1024), ('L',512,1)],
                      'conv_dim': 2048, 'device': get_device(),}
    if net_parameters.get('reduce_label') == True and net_parameters.get('extend_label') == True: # default is Reduce then Extend
        net_parameters['extend_label'] = False
    return net_parameters

def get_optimizer(model: nn.Module, net_parameters: Dict) -> torch.optim.Optimizer:
    opt = net_parameters.get('optimizer')
    inital_lr = net_parameters.get('inital_lr')
    if opt == 'adam':
        return torch.optim.Adam(model.parameters(), lr=inital_lr, amsgrad=False)
    elif opt == 'amsgrad':
        return torch.optim.Adam(model.parameters(), lr=inital_lr, amsgrad=True)

def get_scheduler(optimizer: torch.optim.Optimizer, net_parameters: Dict) -> torch.optim.lr_scheduler._LRScheduler:
    use_scheduler = net_parameters.get('scheduler').get('use_scheduler')
    if use_scheduler:
        scheduler_type = net_parameters.get('scheduler').get('type')
        scheduler_kwargs = net_parameters.get('scheduler').get('kwargs')
        if scheduler_type == 'warm up':
            return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_kwargs)
        elif scheduler_type == 'cycle':
            return torch.optim.lr_scheduler.CyclicLR(optimizer, **scheduler_kwargs)
        elif scheduler_type == 'cosine anneal':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        else:
            raise ValueError('scheduler type must be either warm up, cycle or cosine anneal')
    else:
        return None

def get_nmr_filename(nmr_type: str) -> str:
    if nmr_type == 'H_NMR':
        filename = 'data/1H.nmrshiftdb.data.mol_dict.pickle'
    elif nmr_type == 'C_NMR':
        filename = 'data/13C.nmrshiftdb.data.mol_dict.pickle'
    else:
        raise ValueError('nmr_type must be either H_NMR or C_NMR')
    return filename

def reduce_real_label(real_label: Dict) -> Dict:
    reduced_label = {}
    possible_values = set()
    for key, value in real_label.items():
        if value not in possible_values:
            possible_values.add(value)
            reduced_label[key] = value
    return reduced_label

def calculate_gasteiger_charges(mol: Mol) -> List[float]:
    AllChem.ComputeGasteigerCharges(mol)
    charges = [atom.GetProp("_GasteigerCharge") for atom in mol.GetAtoms()]
    return charges

def group_hydrogens_by_charge(mol: Mol, charges: List[float]=None, round_digit: int=4) -> List[List[int]]:
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        if charges is None:
            charges = calculate_gasteiger_charges(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        equiv_classes = Chem.GetMolFrags(mol, asMols=True)
        equivalent_hydrogens = {}
        for mol_frag in equiv_classes:
            atom_indices = [atom.GetIdx() for atom in mol_frag.GetAtoms() if atom.GetSymbol() == 'H']
            for idx in atom_indices:
                charge = round(float(charges[idx]), round_digit)
                if charge not in equivalent_hydrogens:
                    equivalent_hydrogens[charge] = []
                equivalent_hydrogens[charge].append(idx)
        return list(equivalent_hydrogens.values())
    except:
        return []

def get_required_group_number(mol: Mol, num_distinct_h: int, round_digits: List[int]=[2, 3, 4, 5, 6, 7, 8]) -> Union[int, bool]:
    charges = calculate_gasteiger_charges(mol)
    for round_digit in round_digits:
        current_charges = [round(float(charge), round_digit) for charge in charges]
        equivalent_groups = group_hydrogens_by_charge(mol, current_charges, round_digit)
        if len(equivalent_groups) >= num_distinct_h:
            return round_digit
    return False

def extend_real_label(real_label: Dict, equivalent_groups: List[List[int]]) -> Dict:
    extended_label={}
    for key, value in real_label.items():
        equivalent_group = [group for group in equivalent_groups if key in group]
        if len(equivalent_group) > 0:
            for atom_idx in equivalent_group[0]:
                extended_label[atom_idx]=value
        else:
            extended_label[key]=value
    return extended_label

def get_reduced_and_extended_label(mol: Mol, real_label: Dict, round_digit:int=4) -> Tuple[Dict, Dict]:
    reduced_label = reduce_real_label(real_label)
    charges = calculate_gasteiger_charges(mol)
    hydrogen_groups = group_hydrogens_by_charge(mol, charges, round_digit)
    extended_label = extend_real_label(reduced_label, equivalent_groups=hydrogen_groups)
    return reduced_label, extended_label

def reduce_nmr_label_df(nmr_label_df: pd.DataFrame) -> pd.DataFrame:
    q = Chem.rdqueries.AtomNumEqualsQueryAtom(1)
    new_df = nmr_label_df.copy()
    new_df[NativeColumnNames.Y_DATA.value] = new_df[NativeColumnNames.Y_DATA.value].apply(reduce_real_label)
    new_df['hs_num'] = new_df[NativeColumnNames.X_DATA.value].apply(lambda x: len(x.GetAtomsMatchingQuery(q)))
    new_df['num_distinct_labels'] = new_df[NativeColumnNames.Y_DATA.value].apply(len)
    return new_df

def extend_nmr_label_df(nmr_label_df: pd.DataFrame) -> pd.DataFrame:
    copy_df = nmr_label_df.copy()
    new_df = reduce_nmr_label_df(copy_df)
    # new_df['required_round_digit'] = new_df.apply(lambda x: get_required_group_number(x.rdmol, x.num_distinct_labels), axis=1)
    new_df['required_round_digit'] = \
         new_df.apply(lambda x: get_required_group_number(x[NativeColumnNames.X_DATA.value], x.num_distinct_labels), axis=1)
    extendable_mask = new_df['required_round_digit'] != False
    extendable_df = new_df[extendable_mask].copy()
    # extendable_df['equivalent_groups'] = extendable_df.apply(lambda x: group_hydrogens_by_charge(x.rdmol, round_digit=x.required_round_digit), axis=1)
    extendable_df['equivalent_groups'] = extendable_df.apply(lambda x: group_hydrogens_by_charge(x[NativeColumnNames.X_DATA.value], round_digit=x.required_round_digit), axis=1)
    extendable_df['extended_label'] = extendable_df.apply(lambda x: extend_real_label(x.value, x.equivalent_groups), axis=1)
    copy_df.loc[extendable_mask, NativeColumnNames.Y_DATA.value] = extendable_df['extended_label']
    return copy_df

def shuffle_train_test_df(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    full_df = pd.concat([train_df, test_df])
    random_train_choich = np.random.choice(full_df.index, size=len(train_df.index))
    shuffled_train_df = full_df.loc[random_train_choich] # maybe need to reset index or iloc
    shuffled_test_df = full_df.drop(random_train_choich)
    return shuffled_train_df, shuffled_test_df

def load_pickle_with_mol_dict(filename: str) -> Dict:
    loaded_data = pickle.load(open(filename, 'rb'))
    return loaded_data

def dump_pickle_with_mol_dict(filename: str, data: Dict) -> None:
    pickle.dump(data, open(filename, 'wb'))

def import_data(data_filename: str, extend_label: bool=False, reduce_label: bool=True, shuffle_labels: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loaded_dict = load_pickle_with_mol_dict(data_filename)
    train_df = loaded_dict.get('train_df')
    test_df = loaded_dict.get('test_df')
    train_df['value'] = train_df['value'].apply(lambda x: x[0])
    test_df['value'] = test_df['value'].apply(lambda x: x[0])
    if shuffle_labels:
        train_df, test_df = shuffle_train_test_df(train_df, test_df)
    if reduce_label:
        train_df = reduce_nmr_label_df(train_df)
        test_df = reduce_nmr_label_df(test_df)
    if extend_label:
        train_df = extend_nmr_label_df(train_df)
        test_df = extend_nmr_label_df(test_df)
    return train_df, test_df

def import_data_by_net_parameteres(net_parameteres, allow_shuffle=False, return_lists=True):
    h_nmr_filename = get_nmr_filename(net_parameteres.get('nmr_type'))
    if net_parameteres.get('shuffle_labels') and not allow_shuffle:
        print('shuffle labels is turned off')
        shuffle_labels = False
    else:
        shuffle_labels = net_parameteres.get('shuffle_labels')
    train_df, test_df = import_data(h_nmr_filename, extend_label=net_parameteres.get('extend_label'),
                                    reduce_label=net_parameteres.get('reduce_label'), shuffle_labels=shuffle_labels)
    if return_lists:
        train_data = train_df[NativeColumnNames.X_DATA.value].tolist()
        train_labels = train_df[NativeColumnNames.Y_DATA.value].tolist()
        test_data = test_df[NativeColumnNames.X_DATA.value].tolist()
        test_labels = test_df[NativeColumnNames.Y_DATA.value].tolist()
        return train_data, train_labels, test_data, test_labels
    else:
        return train_df, test_df

def onek_encoding_unk(x: Any, allowable_set: List) -> List[int]:
    encoded_vector = []
    for item in allowable_set:
        if x == item:
            encoded_vector.append(1)
        else:
            encoded_vector.append(0)
    return encoded_vector

def to_device(tensor: torch.Tensor, cuda=True) -> torch.Tensor:
    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()

def list_of_tuples_to_lists(list_of_tuples: List[Tuple]) -> Tuple[List, List]:
    list1, list2 = map(list, zip(*list_of_tuples))
    return list1, list2

def get_graphs_snorm(graphs: dgl.DGLGraph, norm_type: str='n') -> torch.Tensor:
    if norm_type == 'n':
        sizes = [graph.number_of_nodes() for graph in graphs]
    elif norm_type == 'e':
            sizes = [graph.number_of_edges() for graph in graphs]
    snorm = [torch.FloatTensor(size, 1).fill_(1 / size) for size in sizes]
    snorm = torch.cat(snorm).sqrt()
    return snorm

def collate(samples: List[Tuple]) -> Tuple[dgl.DGLGraph, List, torch.Tensor, torch.Tensor]:
    graphs, labels = list_of_tuples_to_lists(samples)
    snorm_n = get_graphs_snorm(graphs, norm_type='n')
    snorm_e = get_graphs_snorm(graphs, norm_type='e')
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels, snorm_n, snorm_e

@torch.no_grad()
def get_test_loss(model, criterion, dl_test, device, min_label, max_label, use_std=False):
    model.eval()

    with torch.no_grad():
        running_test_loss = 0.0
        running_abs_error = 0.0
        all_max_errors = []
        for (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in dl_test:
            # move batch to device
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['features'].to(device)
            batch_e = batch_graphs.edata['features'].to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = torch.cat(batch_labels, dim=0).to(device)
            batch_labels = scale_labels(batch_labels, min_label, max_label)

            mu, std = model(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            labels_flat = batch_labels.reshape(-1, 1)

            # compute metrics
            # mask = (labels_flat != float('inf'))
            mask = (labels_flat != Masks.MISSING_LABEL.value)

            if not use_std:
                loss = criterion(labels_flat[mask], mu[mask])
            else:
                loss = criterion(mu[mask], labels_flat[mask], std[mask])


            abs_error = torch.abs(labels_flat[mask] - mu[mask])
            abs_error = abs_error * (max_label - min_label)
            running_test_loss += loss.item()
            running_abs_error += torch.mean(abs_error)
            all_max_errors.append(torch.max(abs_error))

    avg_abs_error_test = running_abs_error / len(dl_test)
    max_abs_error_test = torch.max(torch.stack(all_max_errors))
    test_loss = running_test_loss / len(dl_test)
    if use_std:
        train_mean_std = torch.mean(std)
        train_max_std = torch.max(std)
        train_min_std = torch.min(std)
        return test_loss, avg_abs_error_test, max_abs_error_test, train_mean_std, train_max_std, train_min_std
    else:
        return test_loss, avg_abs_error_test, max_abs_error_test, 0,0,0

def get_tensor_item(tensor: torch.Tensor) -> Any:
    if torch.is_tensor(tensor):
        return tensor.item()
    else:
        return tensor

def to_tensorboard(writer: SummaryWriter, metrics: dict, epoch: int, use_std: bool) -> None:
    writer.add_scalar(TENSORBOARD_SCALARS.TRAIN_LOSS.value, get_tensor_item(metrics.get('train_loss')), epoch)
    writer.add_scalar(TENSORBOARD_SCALARS.TEST_LOSS.value, get_tensor_item(metrics.get('test_loss')), epoch)
    writer.add_scalar(TENSORBOARD_SCALARS.TRAIN_AVG_ABS_ERROR.value, get_tensor_item(metrics.get('train_abs_error')), epoch)
    writer.add_scalar(TENSORBOARD_SCALARS.TRAIN_MAX_ABS_ERROR.value, get_tensor_item(metrics.get('train_max_error')), epoch)
    writer.add_scalar(TENSORBOARD_SCALARS.TEST_AVG_ABS_ERROR.value, get_tensor_item(metrics.get('test_abs_error')), epoch)
    writer.add_scalar(TENSORBOARD_SCALARS.TEST_MAX_ABS_ERROR.value, get_tensor_item(metrics.get('test_max_error')), epoch)

    if use_std:
        writer.add_scalar('training mean std', get_tensor_item(metrics.get('train_mean_std')), epoch)
        writer.add_scalar('training max std', get_tensor_item(metrics.get('train_max_std')), epoch)
        writer.add_scalar('training min std', get_tensor_item(metrics.get('train_min_std')), epoch)
        writer.add_scalar('test mean std', get_tensor_item(metrics.get('test_mean_std')), epoch)
        writer.add_scalar('test max std', get_tensor_item(metrics.get('test_max_std')), epoch)
        writer.add_scalar('test min std', get_tensor_item(metrics.get('test_min_std')), epoch)


def load_training_info_from_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        lines = f.readlines()

    loaded_dict = {}
    for line in lines:
        key, value = line.strip().split(': ')
        loaded_dict[key] = value

    return loaded_dict

def scale_labels(labels, min_label, max_label):
    """Scales labels to be between 0 and 1 using min-max scaling."""
    return (labels - min_label) / (max_label - min_label)

def inverse_scale_labels(scaled_labels, min_label, max_label):
    """Inverse scales labels to their original range."""
    return scaled_labels * (max_label - min_label) + min_label



def print_non_inf(tensor):
    """
    Prints all values in the given tensor that are not 'inf'.

    Args:
    - tensor (torch.Tensor): Input tensor.
    """
    # Create a mask where tensor values are not inf
    mask = tensor != float('inf')

    # Filter and print the values
    filtered_values = tensor[mask]
    print(filtered_values)


if __name__ == '__main__':
    pass
