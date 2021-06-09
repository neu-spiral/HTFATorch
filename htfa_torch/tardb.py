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

def _densify(tr):
    tr['activations'] = tr['activations'].to_dense()
    return tr

class FmriTarDataset:
    def __init__(self, path):
        metadata = torch.load(path + '.meta')

        self._dataset = wds.WebDataset(path, length=metadata['num_times']).\
                        decode()
        self.voxel_locations = metadata['voxel_locations']

        self._blocks = self._unique_properties(lambda tr: {
            'id': tr['block.id'],
            'run': tr['run.id'],
            'subject': tr['subject.id'],
            'task': tr['task.txt'],
            'template': tr['template.txt'],
            'times': []
        })
        for tr in self._dataset:
            self.blocks[tr['block.id']]['times'].append(tr['time.index'])

    def _unique_properties(self, key_func, data=None):
        if data is None:
            data = self._dataset

        results = []
        result_set = set()

        for rec in data:
            prop = key_func(rec)
            if prop.__hash__:
                prop_ = prop
            else:
                prop_ = str(prop)
            if prop_ not in result_set:
                results.append(prop)
                result_set.add(prop_)

        return results

    @property
    def blocks(self):
        return self._blocks

    def data(self):
        return self._dataset.rename(
            activations='pth', t='time.index', block='block.id', run='run.id',
            subject='subject.id', task='task.txt', template='template.txt',
            individual_differences='individual_differences.json',
            __key__='__key__').\
        map(_densify)

    def mean_block(self):
        num_times = max(row['time.index'] for row in self._dataset) + 1
        mean = torch.zeros(num_times, self.voxel_locations.shape[0])
        for tr in self._dataset:
            mean[tr['time.index']] += tr['pth'].to_dense()
        return mean / len(self.blocks)

    def normalize_activations(self):
        subject_runs = self.subject_runs()
        run_activations = {(subject, run): [tr['pth'] for tr in self._dataset
                                            if tr['run.id'] == run and\
                                            tr['subject.id'] == subject]
                           for subject, run in subject_runs}
        for sr, acts in run_activations.items():
            run_activations[sr] = torch.stack([act.to_dense() for act in acts],
                                              dim=0).flatten()

        normalizers = []
        sufficient_stats = []
        for block in self.blocks:
            activations = run_activations[(block['subject'], block['run'])]
            normalizers.append(torch.abs(activations).max())
            sufficient_stats.append((torch.mean(activations, dim=0),
                                     torch.std(activations, dim=0)))

        return normalizers, sufficient_stats

    def runs(self):
        return self._unique_properties(lambda b: b['run'], self.blocks)

    def subjects(self):
        return self._unique_properties(lambda b: b['subject'], self.blocks)

    def subject_runs(self):
        return self._unique_properties(lambda b: (b['subject'], b['run']),
                                       self.blocks)

    def tasks(self):
        return self._unique_properties(lambda b: b['task'], self.blocks)

    def templates(self):
        return self._unique_properties(lambda b: b['template'], self.blocks)
