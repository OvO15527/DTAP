import dgl
import pandas as pd
import torch
import numpy as np

from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DTIDataset(Dataset):
    def __init__(self, dataset='Davis', compound_graph=None, compound_id=None, protein_graph=None, protein_embedding=None, protein_id=None, label=None, Compound_embedding=None, label_RMSD=None):

        self.dataset = dataset
        self.compound_graph, _ = load_graphs(compound_graph)
        self.compound_graph = list(self.compound_graph)

        self.protein_graph, _ = load_graphs(protein_graph)
        self.protein_graph = list(self.protein_graph)
        self.protein_embedding = np.load(protein_embedding, allow_pickle=True)

        self.compound_id = np.load(compound_id, allow_pickle=True)
        self.protein_id = np.load(protein_id, allow_pickle=True)
        self.label = np.load(label, allow_pickle=True)
        self.label_RMSD = np.load(label_RMSD, allow_pickle=True)
        self.Compound_embedding = np.load(Compound_embedding, allow_pickle=True)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        compound_len = self.compound_graph[idx].num_nodes()
        protein_len = self.protein_graph[idx].num_nodes()

        return self.compound_graph[idx], self.protein_graph[idx], self.protein_embedding[idx], self.Compound_embedding[idx], compound_len, protein_len, self.label[idx], self.label_RMSD[idx]

    def collate(self, sample):

        compound_graph, protein_graph, protein_embedding, Compound_embedding, compound_len, protein_len, label, label_RMSD = map(list, zip(*sample))

        protein_embedding1 = [torch.FloatTensor(emb) for emb in protein_embedding]
        protein_embedding1 = torch.nn.utils.rnn.pad_sequence(protein_embedding1, batch_first=True)


        Compound_embedding1 = [arr.astype(np.float64) for arr in Compound_embedding]
        Compound_embedding1 = [torch.FloatTensor(embc) for embc in Compound_embedding1]
        Compound_embedding1 = torch.nn.utils.rnn.pad_sequence(Compound_embedding1, batch_first=True)

        compound_graph = dgl.batch(compound_graph).to(device)
        protein_graph = dgl.batch(protein_graph).to(device)
        protein_embedding = torch.FloatTensor(protein_embedding1).to(device)
        Compound_embedding = torch.FloatTensor(Compound_embedding1).to(device)
        label = torch.FloatTensor(label).to(device)
        label_RMSD = torch.FloatTensor(label_RMSD).to(device)
        return compound_graph, protein_graph, protein_embedding, Compound_embedding, label, label_RMSD

