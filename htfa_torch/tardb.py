"""Utilities for topographic factor analysis"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'sennesh.e@husky.neu.edu',
             'khan.zu@husky.neu.edu')
from functools import lru_cache
import json
import logging
import os
import types

import torch
import torch.utils.data
import webdataset as wds

from . import utils

class FmriTarDataset:
    def __init__(self, path):
        self._dataset = wds.WebDataset(path).decode()
        self.voxel_locations = torch.load(path + '.voxel_locs.pth')

    def mean_block(self):
        blocks = set()
        num_times = max(row['time.index'] for row in self._dataset) + 1
        mean = torch.zeros(num_times, self.voxel_locations.shape[0])
        for tr in self._dataset:
            blocks.add(tr['block.id'])
            mean[tr['time.index']] += tr['pth'].to_dense()
        return mean / len(blocks)
