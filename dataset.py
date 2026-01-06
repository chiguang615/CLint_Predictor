import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from pubchemfp import GetPubChemFPs
from torch_geometric.data import Data
from torch_geometric.data import Batch
import networkx as nx
import pickle
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate(data_list):
    return Batch.from_data_list(data_list, follow_batch=['bond_node_attr'])

def mol_to_fp(mol):
    maccs = AllChem.GetMACCSKeysFingerprint(mol)
    arr1 = np.zeros((167,), dtype=np.int8)
    ConvertToNumpyArray(maccs, arr1)

    erg = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=15, minPath=1)
    arr2 = np.array(erg, dtype=np.float32)

    pubchem = GetPubChemFPs(mol)
    arr3 = np.array(pubchem, dtype=np.int8)

    return np.concatenate([arr1, arr2, arr3])

def standardize(mol):
    try:
        clean_mol = rdMolStandardize.Cleanup(mol) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        uncharger = rdMolStandardize.Uncharger() 
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        te = rdMolStandardize.TautomerEnumerator() 
        mol_final = te.Canonicalize(uncharged_parent_clean_mol)
    except:
        try:
            mol_final = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            mol_final = mol

    return mol_final

def dataprocess(path, file):
    data = pd.read_csv(path)
    smiles_raw = data['SMILES'].to_list()
    labels = data['LOG CLint'].to_list()
    std_mols = []
    std_smiles = []
    bad_idx = []
    for i, s in enumerate(smiles_raw):
        try:
            m = Chem.MolFromSmiles(s)
            m = standardize(m)
            m = Chem.AddHs(m)
            std_mols.append(m)
            std_smiles.append(Chem.MolToSmiles(m))
        except Exception as e:
            bad_idx.append(i)

    if bad_idx:
        labels = [y for i, y in enumerate(labels) if i not in bad_idx]

    data_list = []
    for i, (mol, smi) in enumerate(tqdm(zip(std_mols, std_smiles), total=len(std_mols))):
        fp = mol_to_fp(mol)
        atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        c_size, atom_types, atom_features_list, edge_index, edge_attrs = smile_to_graph(mol)

        base_x = torch.tensor(atom_features_list, dtype=torch.float)

        data1 = Data(
            x=base_x,
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            fp=torch.tensor(fp, dtype=torch.float),
            y=torch.tensor([labels[i]], dtype=torch.float),
            smiles=smi,
            z=torch.tensor(atom_numbers, dtype=torch.long),
            edge_attr=edge_attrs
        )
        data1.__setitem__('c_size', torch.LongTensor([c_size]))
        data_list.append(data1)

    with open(file, 'wb') as f:
        pickle.dump(data_list, f)
    return data_list

def smile_to_graph(mol):
    c_size = mol.GetNumAtoms()

    atom_types = []
    atom_features_list = []

    for atom in mol.GetAtoms():
        idx, feats = atom_features(atom)
        atom_types.append(idx)
        atom_features_list.append(feats)

    atom_types = torch.tensor(atom_types, dtype=torch.long).unsqueeze(-1)
    atom_features_list = torch.tensor(atom_features_list, dtype=torch.float)

    edges = []
    edge_attrs = []

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        edge_attr = get_edge_features(bond)
        edges.append([a, b])
        edges.append([b, a])
        edge_attrs.append(edge_attr)
        edge_attrs.append(edge_attr)
    
    edge_attrs = torch.stack(edge_attrs)

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    if edge_index==[]:
        edge_index.append([0,1])

    return c_size, atom_types, atom_features_list, edge_index, edge_attrs

atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca',
             'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb','Sn','Ag',
             'Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In',
             'Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
atom_to_idx = {a:i for i,a in enumerate(atom_list)}

def atom_features(atom):
    atom_type_idx = torch.tensor([atom_to_idx.get(atom.GetSymbol(), atom_to_idx['Unknown'])])

    type = np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr','Pt','Hg','Pb','Unknown']))
    degree = np.array(one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]), dtype=np.float32)
    total_H = np.array(one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6,7,8,9,10]), dtype=np.float32)
    valence = np.array(one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6,7,8,9,10]), dtype=np.float32)
    hybrid = np.array(one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP,
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2,
                        'other'
                    ]), dtype=np.float32)
    aromatic = np.array([atom.GetIsAromatic()], dtype=np.float32)

    other_features = np.concatenate([type, degree, total_H, valence, hybrid, aromatic])

    return atom_type_idx, other_features

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def get_edge_features(bond):
    bond_type_dict = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }
    bond_type = bond_type_dict.get(bond.GetBondType(), 4)
    return torch.tensor([bond_type], dtype=torch.long) 