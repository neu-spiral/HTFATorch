"""Deep factor analysis models as ProbTorch modules"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import collections

import numpy as np
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
                'sigma': torch.ones(self.num_subjects, self.embedding_dim) *\
                         np.sqrt(tfa_models.SOURCE_WEIGHT_STD_DEV**2 +\
                                 tfa_models.SOURCE_LOG_WIDTH_STD_DEV**2),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.embedding_dim) *\
                         tfa_models.SOURCE_WEIGHT_STD_DEV,
            },
            'voxel_noise': torch.ones(1) * tfa_models.VOXEL_NOISE,
        })

        super(self.__class__, self).__init__(params, guide=False)

class DeepTFAGuideHyperparams(tfa_models.HyperParams):
    def __init__(self, num_blocks, num_times, num_factors, num_subjects,
                 num_tasks, hyper_means, embedding_dim=2):
        self.num_blocks = num_blocks
        self.num_subjects = num_subjects
        self.num_tasks = num_tasks
        self.num_times = max(num_times)
        self._num_factors = num_factors
        self.embedding_dim = embedding_dim

        params = utils.vardict({
            'subject': {
                'mu': torch.zeros(self.num_subjects, self.embedding_dim),
                'sigma': torch.ones(self.num_subjects, self.embedding_dim),
            },
            'task': {
                'mu': torch.zeros(self.num_tasks, self.embedding_dim),
                'sigma': torch.ones(self.num_tasks, self.embedding_dim),
            },
            'factor_centers': {
                'mu': hyper_means['factor_centers'].expand(self.num_subjects,
                                                           self._num_factors,
                                                           3),
                'sigma': torch.ones(self.num_subjects, self._num_factors, 3),
            },
            'factor_log_widths': {
                'mu': hyper_means['factor_log_widths'].expand(
                    self.num_subjects, self._num_factors
                ),
                'sigma': torch.ones(self.num_subjects, self._num_factors) *\
                         hyper_means['factor_log_widths'].std(),
            },
            'weights': {
                'mu': torch.zeros(self.num_blocks, self.num_times,
                                  self._num_factors),
                'sigma': torch.ones(self.num_blocks, self.num_times,
                                    self._num_factors),
            },
        })

        super(self.__class__, self).__init__(params, guide=True)

class DeepTFADecoder(nn.Module):
    """Neural network module mapping from embeddings to a topographic factor
       analysis"""
    def __init__(self, num_factors, hyper_means, embedding_dim=2):
        super(DeepTFADecoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_factors = num_factors

        self.factors_embedding = nn.Sequential(
            nn.Linear(self._embedding_dim, self._embedding_dim * 2),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 2, self._embedding_dim * 4),
            nn.PReLU(),
            nn.Linear(self._embedding_dim * 4, self._num_factors * 4 * 2),
        )
        factor_bias_loc = torch.cat(
            (hyper_means['factor_centers'],
             hyper_means['factor_log_widths'].unsqueeze(-1)),
            dim=-1
        )
        factor_bias_scale = torch.cat(
            (0.5 * hyper_means['factor_centers'].std(dim=0).expand(
                self._num_factors, 3
            ), hyper_means['factor_log_widths'].std().expand(
                self._num_factors, 1
            )),
            dim=-1
        )
        self.factors_embedding[-1].bias = nn.Parameter(
            torch.stack((factor_bias_loc, factor_bias_scale), dim=-1).reshape(
                self._num_factors * 4 * 2
            )
        )
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
            sigma = predictions.select(-1, 1)
        else:
            mu = params[param]['mu']
            sigma = params[param]['sigma']
            if index is None:
                mu = mu.mean(dim=1)
                sigma = sigma.mean(dim=1)
            else:
                if isinstance(index, tuple):
                    for i in index:
                        mu = mu.select(1, i)
                        sigma = sigma.select(1, i)
                else:
                    mu = mu[:, index]
                    sigma = sigma[:, index]
        result = trace.normal(mu, softplus(sigma),
                              value=utils.clamped(name, guide), name=name)
        return result

    def predict(self, trace, params, guide, subject, task, times=(0, 1),
                block=-1, generative=False):
        origin = torch.zeros(params['subject']['mu'].shape[0],
                             self._embedding_dim)
        origin = origin.to(params['subject']['mu'])
        if subject is not None:
            subject_embed = self._predict_param(
                params, 'subject', subject, None,
                'z^P_{%d,%d}' % (subject, block), trace, False, guide
            )
        else:
            subject_embed = origin
        if task is not None:
            task_embed = self._predict_param(
                params, 'task', task, None, 'z^S_{%d,%d}' % (task, block),
                trace, False, guide
            )
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
            -1, times[1]-times[0], self._num_factors, 2
        )

        centers_predictions = self._predict_param(
            params, 'factor_centers', subject, centers_predictions,
            'FactorCenters%d' % block, trace, predict=generative,
            guide=guide,
        )
        log_widths_predictions = self._predict_param(
            params, 'factor_log_widths', subject, log_widths_predictions,
            'FactorLogWidths%d' % block, trace, predict=generative,
            guide=guide,
        )
        weight_predictions = self._predict_param(
            params, 'weights', block, weight_predictions,
            'Weights%d_%d-%d' % (block, times[0], times[1]), trace,
            predict=generative or block < 0, guide=guide,
        )

        return centers_predictions, log_widths_predictions, weight_predictions

    def forward(self, trace, blocks, block_subjects, block_tasks, params, times,
                guide=None, num_particles=tfa_models.NUM_PARTICLES,
                generative=False):
        params = utils.vardict(params)
        if generative:
            for k, v in params.items():
                params[k] = v.expand(num_particles, *v.shape)

        if blocks:
            weights = [None for b in blocks]
            factor_centers = [None for b in blocks]
            factor_log_widths = [None for b in blocks]

            for (i, b) in enumerate(blocks):
                subject = block_subjects[i] if b is not None else None
                task = block_tasks[i] if b is not None else None

                factor_centers[i], factor_log_widths[i], weights[i] =\
                    self.predict(trace, params, guide, subject, task, times, b,
                                 generative)
        else:
            subject = block_subjects[0] if block_subjects else None
            task = block_tasks[0] if block_tasks else None
            factor_centers, factor_log_widths, weights =\
                self.predict(trace, params, guide, subject, task, times,
                             generative=generative)

        return weights, factor_centers, factor_log_widths

class DeepTFAGuide(nn.Module):
    """Variational guide for deep topographic factor analysis"""
    def __init__(self, num_factors, block_subjects, block_tasks, num_blocks=1,
                 num_times=[1], embedding_dim=2, hyper_means=None):
        super(self.__class__, self).__init__()
        self._num_blocks = num_blocks
        self._num_times = num_times
        self._num_factors = num_factors
        self._embedding_dim = embedding_dim

        self.block_subjects = block_subjects
        self.block_tasks = block_tasks
        num_subjects = len(set(self.block_subjects))
        num_tasks = len(set(self.block_tasks))

        self.hyperparams = DeepTFAGuideHyperparams(self._num_blocks,
                                                   self._num_times,
                                                   self._num_factors,
                                                   num_subjects, num_tasks,
                                                   hyper_means,
                                                   embedding_dim)

    def forward(self, decoder, trace, times=None, blocks=None,
                num_particles=tfa_models.NUM_PARTICLES):
        params = self.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.expand(num_particles, *v.shape)
        if blocks is None:
            blocks = list(range(self._num_blocks))

        block_subjects = [self.block_subjects[b]
                          for b in range(self._num_blocks)
                          if b in blocks]
        block_tasks = [self.block_tasks[b] for b in range(self._num_blocks)
                       if b in blocks]
        if times:
            for k, v in params['weights'].items():
                params['weights'][k] = v[:, :, times[0]:times[1], :]

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
        self.likelihoods = [tfa_models.TFAGenerativeLikelihood(
            locations, self._num_times[b], block=b, register_locations=False
        ) for b in range(self._num_blocks)]
        for b, block_likelihood in enumerate(self.likelihoods):
            self.add_module('likelihood' + str(b), block_likelihood)

    def forward(self, decoder, trace, times=None, guide=probtorch.Trace(),
                observations=[], blocks=None):
        params = self.hyperparams.state_vardict()
        if times is None:
            times = (0, max(self._num_times))
        if blocks is None:
            blocks = list(range(self._num_blocks))

        block_subjects = [self.block_subjects[b]
                          for b in range(self._num_blocks)
                          if b in blocks]
        block_tasks = [self.block_tasks[b] for b in range(self._num_blocks)
                       if b in blocks]

        weights, centers, log_widths = decoder(trace, blocks, block_subjects,
                                               block_tasks, params, times,
                                               guide=guide,
                                               num_particles=1,
                                               generative=True)

        return [self.likelihoods[b](trace, weights[i], centers[i],
                                    log_widths[i], params, times=times,
                                    observations=observations[i])
                for (i, b) in enumerate(blocks)]
