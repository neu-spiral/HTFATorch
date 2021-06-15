"""Deep factor analysis models as ProbTorch modules"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import collections

import numpy as np
import scipy
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import softplus
import torch.utils.data

import probtorch

from . import htfa_models
from . import tfa_models
from . import utils

class DeepTFAGenerativeHyperparams(tfa_models.HyperParams):
    def __init__(self, num_subjects, num_tasks, embedding_dim=2):
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.zeros(self.num_subjects, self.embedding_dim),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'log_sigma': torch.zeros(self.num_tasks, self.embedding_dim),
            },
            'voxel_noise': torch.ones(1) * tfa_models.VOXEL_NOISE,
        })

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_blocks, num_times, num_factors, num_subjects,
                 num_tasks, hyper_means, embedding_dim=2, time_series=True):
        self.num_blocks = num_blocks
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'log_sigma': torch.zeros(self.num_subjects, self.embedding_dim),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'log_sigma': torch.zeros(self.num_tasks, self.embedding_dim),
            },
            'factor_centers': {
                'mu': hyper_means['factor_centers'].expand(self.num_subjects,
                                                           self._num_factors,
                                                           3),
                'log_sigma': torch.zeros(self.num_subjects, self._num_factors,
                                         3),
            },
            'factor_log_widths': {
                'mu': hyper_means['factor_log_widths'].expand(
                    self.num_subjects, self._num_factors
                ),
                'log_sigma': torch.zeros(self.num_subjects, self._num_factors) +\
                             hyper_means['factor_log_widths'].std().log(),
            },
        })
        if time_series:
            params['weights'] = {
                'mu': torch.zeros(self.num_blocks, self.num_times,
                                  self._num_factors),
                'log_sigma': torch.zeros(self.num_blocks, self.num_times,
                                         self._num_factors),
            }

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFADecoder(nn.Module):
    """Neural network module mapping from embeddings to a topographic factor
       analysis"""
    def __init__(self, num_factors, locations, embedding_dim=2,
                 time_series=True, volume=None):
        super(DeepTFADecoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_factors = num_factors
        self._time_series = time_series

        center, center_sigma = utils.brain_centroid(locations)
        center_sigma = center_sigma.sum(dim=1)
        hull = scipy.spatial.ConvexHull(locations)
        coefficient = 1.0
        if volume is not None:
            coefficient = np.cbrt(hull.volume / self._num_factors)

        self.factors_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim, self._embedding_dim * 2),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 2, self._embedding_dim * 4),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 4, self._num_factors * 4 * 2),
        )
        factor_loc = torch.cat(
            (center.expand(self._num_factors, 3),
             torch.ones(self._num_factors, 1) * np.log(coefficient)),
            dim=-1
        )
        factor_log_scale = torch.cat(
            (torch.log(center_sigma / coefficient).expand(
                self._num_factors, 3
            ), torch.zeros(self._num_factors, 1)),
            dim=-1
        )
        self.factors_embedding[-1].bias = nn.Parameter(
            torch.stack((factor_loc, factor_log_scale), dim=-1).reshape(
                self._num_factors * 4 * 2
            )
        )
        if locations is not None:
            self.register_buffer('locations_min',
                                 torch.min(locations, dim=0)[0])
            self.register_buffer('locations_max',
                                 torch.max(locations, dim=0)[0])
        self.weights_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim * 2, self._embedding_dim * 4),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 4, self._embedding_dim * 8),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 8, self._num_factors * 2),
        )

    def _predict_param(self, params, param, index, predictions, name, trace,
                       predict=True, guide=None):
        if name in trace:
            return trace[name].value
        if predict:
            mu = predictions.select(-1, 0)
            log_sigma = predictions.select(-1, 1)
        else:
            mu = params[param]['mu']
            log_sigma = params[param]['log_sigma']
            if index is None:
                mu = mu.mean(dim=1)
                log_sigma = log_sigma.mean(dim=1)
            else:
                if isinstance(index, tuple):
                    for i in index:
                        mu = mu.select(1, i)
                        log_sigma = log_sigma.select(1, i)
                else:
                    mu = mu[:, index]
                    log_sigma = log_sigma[:, index]
        result = trace.normal(mu, torch.exp(log_sigma),
                              value=utils.clamped(name, guide), name=name)
        return result

    def predict(self, trace, params, guide, block, subject, task, times=(0, 1),
                generative=False):
        origin = torch.zeros(params['subject']['mu'].shape[0],
                             self._embedding_dim)
        origin = origin.to(params['subject']['mu'])
        if subject is not None:
            subject_embed = self._predict_param(params, 'subject', subject,
                                                None, 'z^P', trace, False,
                                                guide)
        else:
            subject_embed = origin
        if task is not None:
            task_embed = self._predict_param(params, 'task', task, None, 'z^S',
                                             trace, False, guide)
        else:
            task_embed = origin
        factor_params = self.factors_embedding(subject_embed).view(
            -1, self._num_factors, 4, 2
        )
        centers_predictions = factor_params[:, :, :3]
        log_widths_predictions = factor_params[:, :, 3]

        joint_embed = torch.cat((subject_embed, task_embed), dim=-1)
        weight_predictions = self.weights_embedding(joint_embed).view(
            -1, self._num_factors, 2
        )
        weight_predictions = weight_predictions.unsqueeze(1).expand(
            -1, times.shape[0], self._num_factors, 2
        )

        centers_predictions = self._predict_param(
            params, 'factor_centers', subject, centers_predictions,
            'FactorCenters', trace, predict=generative, guide=guide,
        )
        if 'locations_min' in self._buffers:
            centers_predictions = utils.clamp_locations(centers_predictions,
                                                        self.locations_min,
                                                        self.locations_max)
        log_widths_predictions = self._predict_param(
            params, 'factor_log_widths', subject, log_widths_predictions,
            'FactorLogWidths', trace, predict=generative, guide=guide,
        )
        weight_predictions = self._predict_param(
            params, 'weights', block, weight_predictions,
            'Weights_%d-%d' % (times[0], times[-1]), trace,
            predict=generative or (block < 0).any() or not self._time_series,
            guide=guide,
        )

        return centers_predictions, log_widths_predictions, weight_predictions

    def forward(self, trace, blocks, subjects, tasks, params, times, guide=None,
                num_particles=tfa_models.NUM_PARTICLES, generative=False):
        params = utils.vardict(params)
        if generative:
            for k, v in params.items():
                params[k] = v.expand(num_particles, *v.shape)

        factor_centers, factor_log_widths, weights =\
            self.predict(trace, params, guide, blocks, subjects, tasks, times,
                         generative=generative)

        return weights, factor_centers, factor_log_widths

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_factors, block_subjects, block_tasks, num_blocks=1,
                 num_times=[1], embedding_dim=2, hyper_means=None,
                 time_series=True):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._num_factors = num_factors
        self._embedding_dim = embedding_dim
        self._time_series = time_series

        self.block_subjects = block_subjects
        self.block_tasks = block_tasks
        num_subjects = len(set(self.block_subjects))
        num_tasks = len(set(self.block_tasks))

        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks,
                                                   hyper_means,
                                                   embedding_dim, time_series)

    def forward(self, decoder, trace, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.expand(num_particles, *v.shape)
        if blocks is None:
            blocks = torch.arange(self._num_blocks)

        block_subjects = torch.tensor(self.block_subjects,
                                      dtype=torch.long)[blocks]
        block_tasks = torch.tensor(self.block_tasks, dtype=torch.long)[blocks]
        if times is not None and self._time_series:
            for k, v in params['weights'].items():
                params['weights'][k] = v[:, :, times, :]

        return decoder(trace, blocks, block_subjects, block_tasks, params,
                       times=times, num_particles=num_particles)

class DeepTFAModel(nn.Module):
    """Generative model for deep topographic factor analysis"""
    def __init__(self, locations, block_subjects, block_tasks,
                 num_factors=tfa_models.NUM_FACTORS, num_blocks=1,
                 num_times=[1], embedding_dim=2):
        super(self.__class__, self).__init__()
        self._locations = locations
        self._num_factors = num_factors
        self._num_blocks = num_blocks
        self._num_times = num_times
        self.block_subjects = block_subjects
        self.block_tasks = block_tasks

        self.hyperparams = DeepTFAGenerativeHyperparams(
            len(set(block_subjects)), len(set(block_tasks)), embedding_dim
        )
        self.add_module('likelihood', tfa_models.TFAGenerativeLikelihood(
            locations, self._num_times, block=None, register_locations=False
        ))

    def forward(self, decoder, trace, times=None, guide=None, observations=[],
                blocks=None, locations=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        if guide is None:
            guide = probtorch.Trace()
        if times is None:
            times = torch.arange(max(self._num_times))
        if blocks is None:
            blocks = torch.arange(self._num_blocks)
        else:
            tr_blocks = blocks
            blocks = blocks.unique()

        block_subjects = torch.tensor(self.block_subjects,
                                      dtype=torch.long)[blocks]
        block_tasks = torch.tensor(self.block_tasks, dtype=torch.long)[blocks]

        weights, centers, log_widths = decoder(trace, blocks, block_subjects,
                                               block_tasks, params, times,
                                               guide=guide,
                                               num_particles=num_particles,
                                               generative=True)

        return self.likelihood(trace, weights, centers, log_widths, params,
                               times=times, observations=observations,
                               blocks=tr_blocks, locations=locations)
