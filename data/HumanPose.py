import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

# import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib

# 
from data.data_load import load_data

class HumanPoseDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_dir, mode, config):
        """
            Loading Human3.6M dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        output_filename_3d = os.path.join(data_dir, 'data_3d_h36m.npz')
        output_filename_2d = os.path.join(data_dir, 'data_2d_h36m_gt.npz')
        
        data_input_list, data_label_list = load_data(output_filename_3d, output_filename_2d, mode, config['len_input'], config['len_output'])

        self.input_list = data_input_list
        self.label_list = data_label_list

        assert len(self.input_list) == len(self.label_list)
        
        self.n_samples = len(self.input_list)

        print('{} sizes : {}'.format(mode, self.n_samples))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.input_list[idx], self.label_list[idx]

if __name__ == '__main__':

    dataset = HumanPoseDataset(name='human3.6M', data_dir='./human36', mode='train')

    input, label = dataset[0]

    print("input:", input.shape)
    print("label:", label.shape)