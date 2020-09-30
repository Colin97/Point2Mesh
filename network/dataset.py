# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, root = '../data/all', npoints = 12800, ntriangles = 25000, split = 'train'):
        self.npoints = npoints
        self.ntriangles = ntriangles
        self.root = root
        self.split = split

        if split == 'train':
            with open(os.path.join(self.root, 'train.txt'), 'r') as f:
                self.ids = [line.strip() for line in f.readlines()]
        else:
            with open(os.path.join(self.root, 'val.txt'), 'r') as f:
                self.ids = [line.strip() for line in f.readlines()]

        self.data = []
        for id in self.ids:
            d = pickle.load(open(os.path.join(self.root, id + '.p'), 'rb')) 
            self.data.append(d)

    def __getitem__(self, index):
        d = self.data[index]
        idx = torch.randperm(len(d['vertex_idx']))
        idx = idx[:self.ntriangles]
        return torch.from_numpy(d['pc']).float(), torch.from_numpy(d['vertex_idx'][idx]).long(), torch.from_numpy(d['label'][idx]).long()

    def __len__(self):
        return len(self.ids)

class TestDataset(Dataset):
    def __init__(self, root, npoints=12800, ntriangles=350000):
        self.npoints = npoints
        self.ntriangles = ntriangles
        self.root = root

        with open(os.path.join(self.root, 'models.txt'), 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        self.data = []
        for i, id in enumerate(self.ids):
            d = pickle.load(open(os.path.join(self.root, id + '.p'), 'rb'))
            n = len(d['vertex_idx']) 
            for i in range((n + self.ntriangles - 1) // self.ntriangles):
                dd = {}
                dd['pc'] = d['pc']
                l = i * self.ntriangles
                r = min((i + 1) * self.ntriangles, n)
                dd['vertex_idx'] = d['vertex_idx'][l: r]
                dd['label'] = d['label'][l: r]
                if (r - l < self.ntriangles):
                    padding_size = self.ntriangles - (r - l)
                    ver_padding = np.repeat(np.expand_dims(d['vertex_idx'][-1], 0), padding_size, axis = 0)
                    label_padding = np.repeat(np.expand_dims(d['label'][-1], 0), padding_size, axis = 0)
                    dd['vertex_idx'] = np.concatenate((dd['vertex_idx'], ver_padding), axis=0)
                    dd['label'] = np.concatenate((dd['label'], label_padding), axis=0)
                dd['model_ids'] = id + '_%d'%i
                self.data.append(dd)

    def __getitem__(self, index):
        d = self.data[index]
        return d['pc'], d['vertex_idx'], d['label'], d['model_ids']

    def __len__(self):
        return len(self.data)

