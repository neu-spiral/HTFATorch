"""Perform plain topographic factor analysis on a given fMRI data file."""

__author__ = 'Eli Sennesh', 'Zulqarnain Khan'
__email__ = 'e.sennesh@northeastern.edu', 'khan.zu@husky.neu.edu'

import logging
import pickle
import time

import hypertools as hyp
import nilearn.image
import nilearn.plotting as niplot
import numpy as np
import scipy.io as sio
import scipy.spatial.distance as sd
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.utils.data

import probtorch

from . import utils
from . import tfa_models

# check the availability of CUDA
CUDA = torch.cuda.is_available()

# placeholder values for hyperparameters
LEARNING_RATE = 0.1

EPOCH_MSG = '[Epoch %d] (%dms) Posterior free-energy %.8e'

def free_energy(q, p, num_particles=tfa_models.NUM_PARTICLES):
    """Calculate the free-energy (negative of the evidence lower bound)"""
    if num_particles and num_particles > 0:
        sample_dim = 0
    else:
        sample_dim = None
    return -probtorch.objectives.montecarlo.elbo(q, p, sample_dim=sample_dim)

def log_likelihood(q, p, num_particles=tfa_models.NUM_PARTICLES):
    """The expected log-likelihood of observed data under the proposal distribution"""
    if num_particles and num_particles > 0:
        sample_dim = 0
    else:
        sample_dim = None
    return probtorch.objectives.montecarlo.log_like(q, p, sample_dim=sample_dim)

class TopographicalFactorAnalysis:
    """Overall container for a run of TFA"""
    def __init__(self, data_file, num_factors=tfa_models.NUM_FACTORS):
        self.num_factors = num_factors

        self.voxel_activations, self._image, self.voxel_locations, self._name,\
            self._template = utils.load_dataset(data_file)

        # Pull out relevant dimensions: the number of times-of-recording, and
        # the number of voxels in each timewise "slice"
        self.num_times = self.voxel_activations.shape[0]
        self.num_voxels = self.voxel_activations.shape[1]

        # Estimate further hyperparameters from the dataset
        self.brain_center, self.brain_center_std_dev =\
            utils.brain_centroid(self.voxel_locations)

        mean_centers_init, mean_widths_init, mean_weights_init = \
            utils.initial_hypermeans(self.voxel_activations.t().numpy(),
                                     self.voxel_locations.numpy(),
                                     self.num_factors)
        hyper_means = {
            'factor_centers': torch.Tensor(mean_centers_init),
            'factor_log_widths': mean_widths_init,
            'weights': torch.Tensor(mean_weights_init)
        }
        self.enc = tfa_models.TFAGuide(hyper_means, self.num_times,
                                       num_factors=self.num_factors)

        self.dec = tfa_models.TFAModel(self.brain_center,
                                       self.brain_center_std_dev,
                                       self.num_times, self.voxel_locations,
                                       num_factors=self.num_factors)

        if CUDA:
            self.enc = torch.nn.DataParallel(self.enc)
            self.dec = torch.nn.DataParallel(self.dec)
            self.enc.cuda()
            self.dec.cuda()

    def train(self, num_steps=10, learning_rate=LEARNING_RATE,
              log_level=logging.WARNING, batch_size=64,
              num_particles=tfa_models.NUM_PARTICLES):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        activations_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.voxel_activations,
                torch.zeros(self.voxel_activations.shape)
            ),
            batch_size=batch_size,
            num_workers=2
        )
        optimizer = torch.optim.Adam(list(self.enc.parameters()), lr=learning_rate)

        self.enc.train()
        self.dec.train()

        free_energies = np.zeros(num_steps)
        lls = np.zeros(num_steps)

        for epoch in range(num_steps):
            start = time.time()

            epoch_free_energies = list(range(len(activations_loader)))
            epoch_lls = list(range(len(activations_loader)))
            for (batch, (activations, _)) in enumerate(activations_loader):
                activations = Variable(activations)
                if CUDA:
                    activations = activations.cuda()
                trs = (batch*batch_size, None)
                trs = (trs[0], trs[0] + activations.shape[0])

                optimizer.zero_grad()
                q = probtorch.Trace()
                self.enc(q, times=trs, num_particles=num_particles)
                p = probtorch.Trace()
                self.dec(p, times=trs, guide=q, observations={'Y': activations})

                epoch_free_energies[batch] = free_energy(q, p, num_particles=num_particles)
                epoch_lls[batch] = log_likelihood(q, p, num_particles=num_particles)
                epoch_free_energies[batch].backward()
                optimizer.step()

                if CUDA:
                    epoch_free_energies[batch] = epoch_free_energies[batch].cpu().data.numpy()
                    epoch_lls[batch] = epoch_lls[batch].cpu().data.numpy()

            free_energies[epoch] = np.array(epoch_free_energies).sum(0)
            lls[epoch] = np.array(epoch_lls).sum(0)

            end = time.time()
            msg = EPOCH_MSG % (epoch + 1, (end - start) * 1000, free_energies[epoch])
            logging.info(msg)

        return np.vstack([free_energies, lls])

    def results(self):
        """Return the inferred parameters"""
        q = probtorch.Trace()
        self.enc(q, num_particles=tfa_models.NUM_PARTICLES)

        weights = q['Weights' + str(self.enc.module.subject)].value.data.mean(0)
        factor_centers = q['FactorCenters' + str(self.enc.module.subject)].value.data.mean(0)
        factor_log_widths = q['FactorLogWidths' + str(self.enc.module.subject)].value.data.mean(0)

        if CUDA:
            weights = weights.cpu()
            factor_centers = factor_centers.cpu()
            factor_log_widths = factor_log_widths.cpu()

        factor_centers = factor_centers.numpy()
        factor_log_widths = factor_log_widths.numpy()

        factors = utils.initial_radial_basis(self.voxel_locations.numpy(),
                                             factor_centers,
                                             np.exp(factor_log_widths))

        logging.info('Reconstruction Error (Frobenius Norm): %.8e',
                     np.linalg.norm(weights @ factors -\
                                    self.voxel_activations.numpy()))

        result = {
            'weights': weights.numpy(),
            'factors': factors,
            'factor_centers': factor_centers,
            'factor_log_widths': factor_log_widths,
        }
        return result

    def mean_parameters(self, log_level=logging.WARNING, matfile=None):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)

        if CUDA:
            self.enc.module.hyperparams.cpu()
        params = self.enc.module.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.data

        mean_factor_center = params['factor_centers']['mu'].numpy()
        mean_factor_log_width = params['factor_log_widths']['mu'].numpy()
        mean_weight = params['weights']['mu'].numpy()
        mean_factors = utils.initial_radial_basis(
            self.voxel_locations.numpy(), mean_factor_center,
            np.exp(mean_factor_log_width[0])
        )

        logging.info("Mean Factor Centers: %s", str(mean_factor_center))
        logging.info("Mean Factor Log Widths: %s", str(mean_factor_log_width))
        logging.info("Mean Weights: %s", str(mean_weight))
        logging.info('Reconstruction Error (Frobenius Norm): %.8e',
                     np.linalg.norm(mean_weight @ mean_factors - self.voxel_activations.numpy()))

        mean_parameters = {
            'mean_weight': mean_weight,
            'mean_factor_center': mean_factor_center,
            'mean_factor_log_width': mean_factor_log_width,
            'mean_factors': mean_factors
        }

        if matfile is not None:
            sio.savemat(matfile, mean_parameters, do_compression=True)

        return mean_parameters

    def save(self, out_dir='.'):
        '''Save a TopographicalFactorAnalysis in full to a file for later'''
        with open(out_dir + '/' + self._name + '.tfa', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        '''Load a saved TopographicalFactorAnalysis from a file, saving the
           effort of rerunning inference from scratch.'''
        with open(filename, 'rb') as file:
            return pickle.load(file)

    def plot_voxels(self):
        hyp.plot(self.voxel_locations.numpy(), 'k.')

    def plot_factor_centers(self, filename=None, show=True):
        results = self.results()

        if CUDA:
            self.enc.module.hyperparams.cpu()
        params = self.enc.module.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.data
        uncertainties = params['factor_centers']['sigma']

        plot = niplot.plot_connectome(
            np.eye(self.num_factors),
            results['factor_centers'],
            node_color=utils.uncertainty_palette(uncertainties),
            node_size=np.exp(results['factor_log_widths'] - np.log(2))
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_original_brain(self, filename=None, show=True, plot_abs=False,
                            time=0):
        image = nilearn.image.index_img(self._image, time)
        plot = niplot.plot_glass_brain(image, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction(self, filename=None, show=True, plot_abs=False,
                            log_level=logging.WARNING, time=0):
        results = self.results()
        weights = results['weights']
        factors = results['factors']

        reconstruction = weights @ factors
        image = utils.cmu2nii(reconstruction,
                              self.voxel_locations.numpy(),
                              self._template)
        image_slice = nilearn.image.index_img(image, time)
        plot = niplot.plot_glass_brain(image_slice, plot_abs=plot_abs)

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_connectome(self, filename=None, show=True):
        results = self.results()
        if CUDA:
            self.enc.module.hyperparams.cpu()
        params = self.enc.module.hyperparams.state_vardict()
        for k, v in params.items():
            params[k] = v.data
        uncertainties = params['factor_centers']['sigma']

        connectome = 1 - sd.squareform(sd.pdist(results['weights'].T),
                                       'correlation')
        plot = niplot.plot_connectome(
            connectome,
            results['factor_centers'],
            node_color=utils.uncertainty_palette(uncertainties),
            edge_threshold='75%'
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def sample(self, times=None, posterior_predictive=False):
        q = probtorch.Trace()
        if posterior_predictive:
            self.enc(q, times=times, num_particles=1)
        p = probtorch.Trace()
        self.dec(p, times=times, guide=q, observations=q)
        return p
