import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import random
from numpy.random import permutation


class MultiViewDataset(Dataset):
    def __init__(self, data,label,args):
        self.data = data
        self.label = label
        self.args = args
        self.V = len(data)
        self.N = self.label.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        data = [torch.from_numpy(self.data[v][idx]) for v in range(self.V)]
        true_label = np.int64(self.label[idx])
        return data,true_label,idx



def MultiViewDatasetLoader(args):
    data = []
    mat = sio.loadmat('/home/shanghui/dsh/datasets/MVC/'+args.dataset+'.mat')
    raw_data = mat['X'][0]
    V = raw_data.shape[0]
    scaler = preprocessing.MinMaxScaler()
    for v in range(V):
        data.append(scaler.fit_transform(raw_data[v].astype(np.float32)))

    label = np.squeeze(mat['Y']).astype(np.uint8)
    dim_list = [da.shape[1] for da in data]
    num_sample = label.shape[0]

    dataloader = MultiViewDataset(data,label,args)
    return dataloader, num_sample, dim_list,label
