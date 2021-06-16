"""Utilities for topographic factor analysis"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'sennesh.e@husky.neu.edu',
             'khan.zu@husky.neu.edu')
import torch
import torch.utils.data
import webdataset as wds

def _collation_fn(samples):
    result = {'__key__': [], 'activations': [], 't': [], 'block': []}
    for sample in samples:
        for k, v in sample.items():
            result[k].append(v)

    result['activations'] = torch.stack(result['activations'], dim=0)
    result['t'] = torch.tensor(result['t'], dtype=torch.long)
    result['block'] = torch.tensor(result['block'], dtype=torch.long)

    return result

class FmriTarDataset:
    def __init__(self, path):
        metadata = torch.load(path + '.meta')
        self._path = path
        self._num_times = metadata['num_times']

        self._dataset = wds.WebDataset(path, length=self._num_times)
        self._dataset = self._dataset.decode().rename(
            activations='pth', t='time.index', block='block.id',
            __key__='__key__'
        )
        self._dataset = self._dataset.map_dict(
            activations=lambda acts: acts.to_dense()
        )
        self.voxel_locations = metadata['voxel_locations']

        self._blocks = {}
        for block in metadata['blocks']:
            self._blocks[block['block']] = {
                'id': block['block'],
                'run': block['run'],
                'subject': block['subject'],
                'task': block['task'],
                'template': block['template'],
                'times': []
            }
        for tr in self._dataset:
            self.blocks[tr['block']]['times'].append(tr['t'])

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

    def data(self, batch_size=None, selector=None):
        result = self._dataset
        if batch_size:
            result = result.batched(batch_size, _collation_fn)
        if selector:
            result = result.select(selector)
        return result.dbcache(self._path + '.db', self._num_times)

    def mean_block(self):
        num_times = max(row['t'] for row in self._dataset) + 1
        mean = torch.zeros(num_times, self.voxel_locations.shape[0])
        for tr in self._dataset:
            mean[tr['t']] += tr['activations']
        return mean / len(self.blocks)

    def normalize_activations(self):
        subject_runs = self.subject_runs()
        run_activations = {(subject, run): [tr['activations'] for tr in self._dataset
                                            if self.blocks[tr['block']]['run'] == run and\
                                            self.blocks[tr['block']]['subject'] == subject]
                           for subject, run in subject_runs}
        for sr, acts in run_activations.items():
            run_activations[sr] = torch.stack(acts, dim=0).flatten()

        normalizers = []
        sufficient_stats = []
        for block in self.blocks.values():
            activations = run_activations[(block['subject'], block['run'])]
            normalizers.append(torch.abs(activations).max())
            sufficient_stats.append((torch.mean(activations, dim=0),
                                     torch.std(activations, dim=0)))

        return normalizers, sufficient_stats

    def runs(self):
        return self._unique_properties(lambda b: b['run'], self.blocks.values())

    def subjects(self):
        return self._unique_properties(lambda b: b['subject'],
                                       self.blocks.values())

    def subject_runs(self):
        return self._unique_properties(lambda b: (b['subject'], b['run']),
                                       self.blocks.values())

    def tasks(self):
        return self._unique_properties(lambda b: b['task'],
                                       self.blocks.values())

    def templates(self):
        return self._unique_properties(lambda b: b['template'],
                                       self.blocks.values())
