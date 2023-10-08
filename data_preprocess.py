import torch
from rdkit import Chem
import dgl
from dgl import DGLGraph

from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

from utils import import_data, onek_encoding_unk, get_nmr_filename, Masks, NativeColumnNames
from rdkit.Chem.rdchem import Mol, Atom, Bond
from typing import List, Dict, Tuple
from enum import Enum

class FeaturesConstansts(Enum):
    ELEMENTS = ['H', 'C', 'O', 'N', 'P', 'S', 'F', 'Cl']
    HYBRIDIZATIONS = [Chem.HybridizationType.S,
                      Chem.HybridizationType.SP,
                      Chem.HybridizationType.SP2,
                      Chem.HybridizationType.SP3,
                      Chem.HybridizationType.SP3D,
                      Chem.HybridizationType.SP3D2]
    FORMAL_CHARGES = [-1, 0, 1]
    DEFAULT_VALENCE = range(1, 7)
    RINGS = range(3, 8)
    DEGREES = range(6)
    BONDS = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

def get_rdkit_basefeatures(mol: Mol) -> List:
    from rdkit.Chem import ChemicalFeatures
    fdef_name = 'data/BaseFeatures.fdef'
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_features = mol_featurizer.GetFeaturesForMol(mol)
    return mol_features

def get_donor_acceptor_features(mol: Mol) -> Dict:
    num_atoms = mol.GetNumAtoms()
    is_donor_acceptor = {atom_index : [0]*2 for atom_index in range(num_atoms)}
    mol_feats = get_rdkit_basefeatures(mol)
    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == 'Donor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor_acceptor[u] = [1, 0]
        elif mol_feats[i].GetFamily() == 'Acceptor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor_acceptor[u] = [0, 1]
    return is_donor_acceptor

def append_and_record_feature(current_feature_vector: List, feature_to_add: List, feature_name: str) -> Tuple[List, Dict]:
    start_index = len(current_feature_vector)
    final_feature_vector = current_feature_vector + feature_to_add
    end_index = len(final_feature_vector) - 1
    feature_meta_data = {feature_name: (start_index, end_index)}
    return  final_feature_vector, feature_meta_data
    
def get_atom_features(atom: Atom, features_instructions: Dict, donor_acceptor_features_dict: Dict=None) -> Tuple[List, Dict]:
    metadata = dict()
    pt = Chem.GetPeriodicTable()
    atomic_num = atom.GetAtomicNum()
    ELEM_LIST = FeaturesConstansts.ELEMENTS.value
    HYBRIDIZATIONS = FeaturesConstansts.HYBRIDIZATIONS.value
    FORMAL_CHARGES = FeaturesConstansts.FORMAL_CHARGES.value
    DEFAULT_VALENCE = FeaturesConstansts.DEFAULT_VALENCE.value
    RINGS = FeaturesConstansts.RINGS.value
    DEGREES = FeaturesConstansts.DEGREES.value
    if donor_acceptor_features_dict is None:
        donor_acceptor_features_dict = {}
    donor_acceptor_features = donor_acceptor_features_dict.get(atom.GetIdx(), [])
    atomic_vector = []
    if features_instructions.get('atomic_number', False): # True: # atomic number
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, [atomic_num],
                                                                     'atomic number')
        metadata.update(feature_metadata)
    if features_instructions.get('atomic_number_ohk', False): # True: # atomic number ohk
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, onek_encoding_unk(atom.GetSymbol(), ELEM_LIST),
                                                                     'atomic number ohk')
        metadata.update(feature_metadata)
    if features_instructions.get('valence', False): # True: # valence
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, [atom.GetTotalValence()],
                                                                     'valence')
        metadata.update(feature_metadata)
    if features_instructions.get('valence_ohk', False): # True: # valence ohk
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, onek_encoding_unk(atom.GetTotalValence(), range(1, 7)),
                                                                     'valence ohk')
        metadata.update(feature_metadata)
    if features_instructions.get('aromatic', False): # True: # aromatic
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, [atom.GetIsAromatic()],
                                                                     'aromatic')
        metadata.update(feature_metadata) 
    if features_instructions.get('hybridization_ohk', False): # True: # hybridizations
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, onek_encoding_unk(atom.GetHybridization(), HYBRIDIZATIONS),
                                                                     'hybridization ohk')
        metadata.update(feature_metadata)  
    if features_instructions.get('formal_charge_ohk', False): # True: # formal charge
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGES),
                                                                     'formal charge ohk')
        metadata.update(feature_metadata)       
    if features_instructions.get('default_valence_ohk', False): # True: # Default valence ohk
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, onek_encoding_unk(pt.GetDefaultValence(atomic_num), DEFAULT_VALENCE),
                                                                     'default valence ohk')
        metadata.update(feature_metadata)
    if features_instructions.get('ring_ohk', False): # True: # in ring size ohk
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, [atom.IsInRingSize(r) for r in RINGS],
                                                                     'in ring size ohk')
        metadata.update(feature_metadata)
    if features_instructions.get('donor_acc_ohk', False): # True: # donor-acceptor
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, donor_acceptor_features,
                                                                     'donor acc ohk')
        metadata.update(feature_metadata)
    if features_instructions.get('degree_ohk', False): # False: # degree
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, onek_encoding_unk(atom.GetDegree(), DEGREES),
                                                                     'degree ohk')
        metadata.update(feature_metadata)
    if features_instructions.get('chiral_tag', False):
        pass # str(atom.GetChiralTag())
    if features_instructions.get('num_H_connected', False):
        pass # atom.GetTotalNumHs()
    if features_instructions.get('partial_charge', False):
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, [atom.GetProp('_GasteigerCharge')], 'partial charge')
        metadata.update(feature_metadata)
    if features_instructions.get('r_covalent', False):
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, [pt.GetRcovalent(atomic_num)], 'r_covalent')
        metadata.update(feature_metadata)
    if features_instructions.get('r_vanderwals', False):
        atomic_vector, feature_metadata = append_and_record_feature (atomic_vector, [pt.GetRvdw(atomic_num)], 'r_vanderwals')
        metadata.update(feature_metadata)
    if features_instructions.get('xyz_positions', False):
        pass
    return atomic_vector, metadata

def get_bond_features(bond: Bond, features_instructions: Dict) -> Tuple[List, Dict]:
    metadata = dict()
    bond_vector = []
    possible_bonds = FeaturesConstansts.BONDS.value
    if features_instructions.get('bond_order', False): # bond order
        bond_vector, feature_metadata = append_and_record_feature(bond_vector, onek_encoding_unk(bond.GetBondType(), possible_bonds),
                                                                  'bond order ohk')
        metadata.update(feature_metadata)
    if features_instructions.get('in_ring', False): # in ring
        bond_vector, feature_metadata = append_and_record_feature(bond_vector, [bond.IsInRing()],
                                                                  'bond in ring')
        metadata.update(feature_metadata) 
    return bond_vector, metadata

def get_dgl_plain_graph(mol: Mol, max_nodes:int=64) -> DGLGraph:
    edge_src = []
    edge_dst = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        begin_idx = a1.GetIdx()
        end_idx = a2.GetIdx()
        edge_src.extend([begin_idx, end_idx])
        edge_dst.extend([end_idx, begin_idx])
    g = dgl.graph((edge_src, edge_dst), num_nodes=max_nodes)
    return g

def encode_node_features(graph: DGLGraph, mol: Mol, node_features_instructions: Dict, max_nodes: int) -> Tuple[DGLGraph, Dict]:
    donor_acceptor_features_dict = get_donor_acceptor_features(mol) if node_features_instructions.get('donor_acc_ohk', False) else None
    nodeF = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        atom_features, atom_features_metadata = get_atom_features(atom, node_features_instructions, donor_acceptor_features_dict)
        nodeF.append(atom_features)
    atom_features_len=len(nodeF[0]) 
    n_atoms = mol.GetNumAtoms()
    for _ in range(n_atoms, max_nodes): # add padding
        nodeF.append([0] * atom_features_len)
    graph.ndata['features'] = torch.Tensor(nodeF)
    return graph, atom_features_metadata

def encode_edge_features(graph: DGLGraph, mol: Mol, edges_features_instructions: Dict) -> Tuple[DGLGraph, Dict]:
    edge_src = graph.edges()[0].tolist()
    edge_dst = graph.edges()[0].tolist()
    edgeF = []
    for i, bond in enumerate(mol.GetBonds()):
        bond_features, bond_features_metadata = get_bond_features(bond, edges_features_instructions)
        edgeF.extend([torch.Tensor(bond_features), torch.Tensor(bond_features)])
    graph.edata['features'] = torch.stack(edgeF)
    return graph, bond_features_metadata

def mol2dgl(mol: Mol, max_nodes: int, atom_features_instructions: Dict, bond_features_instructions: Dict) -> Dict:
    g = get_dgl_plain_graph(mol, max_nodes)
    g, atom_features_metadata = encode_node_features(g, mol, atom_features_instructions, max_nodes)
    g, bond_features_metadata = encode_edge_features(g, mol, bond_features_instructions)
    metadata = {'atom_feature_metadata': atom_features_metadata, 'bond_feature_metadata': bond_features_metadata}
    graph_dict_metadata = {'graph': g, 'meta data': metadata}
    return graph_dict_metadata    # n_atoms = mol.GetNumAtoms()

def label_dict_to_pred_tensor(label_dict: Dict, max_nodes: int) -> torch.Tensor:
    label_tensor = Masks.MISSING_LABEL.value * torch.ones(max_nodes)
    for atom_index, value in label_dict.items():
        label_tensor[atom_index] = value
    return label_tensor

class MoleculeDataset(DGLDataset):
    # This should output a data structure with a sparse matrix in it
    def __init__(self, nmr_type, data, labels, atom_features_instructions, bond_features_instructions, MAX_NUM_ATOMS=64, MAX_NUM_BONDS=50):
        super().__init__(name=nmr_type)
        self.data = data
        self.labels = labels
        self.atom_features_instructions = atom_features_instructions
        self.bond_features_instructions = bond_features_instructions
        self.MAX_NUM_ATOMS = MAX_NUM_ATOMS
        self.MAX_NUM_BONDS = MAX_NUM_BONDS
        self.len = len(data)

    def __getitem__(self, index):
        rdmol = self.data[index]
        graph_dict_metadata = mol2dgl(rdmol, self.MAX_NUM_ATOMS, self.atom_features_instructions, self.bond_features_instructions)
        graph = graph_dict_metadata.get('graph')
        self.metadata = graph_dict_metadata.get('meta data')
        label = self.labels[index]
        label_tensor = label_dict_to_pred_tensor(label, self.MAX_NUM_ATOMS)
        return graph, label_tensor

    def __len__(self):
        return len(self.data)

def get_features_instructions(features_set):
    atom_features_instructions, bond_features_instructions = {}, {}
    if features_set==1:
        atom_features_instructions = {'atomic_number': True, 'atomic_number_ohk' : True, 'valence': True, 'valence_ohk': True,
                                      'aromatic': True, 'hybridization_ohk': True, 'formal_charge_ohk': True,
                                      'default_valence_ohk': True, 'ring_ohk': True, 'donor_acc_ohk': False, 'degree_ohk': False,
                                      'chiral_tag': False, 'num_H_connected': False, 'xyz_positions': False}
        bond_features_instructions = {'bond_order': True, 'is aromatic': True}
    return atom_features_instructions, bond_features_instructions    

def get_datasets(nmr_type='H_NMR', max_atoms=64, max_bonds=50, features_set=1, extend_label=True, reduce_label=False, shuffle_labels=False):
    nmr_filename = get_nmr_filename(nmr_type)
    train_df, test_df = import_data(nmr_filename, extend_label=extend_label, reduce_label=reduce_label, shuffle_labels=shuffle_labels)
    atom_features_instructions, bond_features_instructions = get_features_instructions(features_set)
    train = MoleculeDataset(nmr_type, train_df[NativeColumnNames.X_DATA.value].tolist(), train_df[NativeColumnNames.Y_DATA.value].tolist(),
                            atom_features_instructions, bond_features_instructions, max_atoms, max_bonds)
    test = MoleculeDataset(nmr_type, test_df[NativeColumnNames.X_DATA.value].tolist(), test_df[NativeColumnNames.Y_DATA.value].tolist(),
                           atom_features_instructions, bond_features_instructions, max_atoms, max_bonds)
    _ = train.__getitem__(0) # to generate meta-data
    _ = test.__getitem__(0) # to generate meta-data
    return train, test

def get_dataloaders(train_dataset, test_dataset, batch_size=16, shuffle=True, collate=None):
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    pass