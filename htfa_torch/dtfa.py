"""Perform deep topographic factor analysis on fMRI data"""

__author__ = ('Jan-Willem van de Meent',
              'Eli Sennesh',
              'Zulqarnain Khan')
__email__ = ('j.vandemeent@northeastern.edu',
             'e.sennesh@northeastern.edu',
             'khan.zu@husky.neu.edu')

import collections
import datetime
import logging
import os
import pickle
import time

try:
    if __name__ == '__main__':
        import matplotlib
        matplotlib.use('TkAgg')
finally:
    import matplotlib.pyplot as plt
import nilearn.image
import nilearn.plotting as niplot
import numpy as np
import scipy.io as sio
import torch
import torch.distributions as dists
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.functional import softplus
import torch.optim.lr_scheduler

import probtorch

from . import dtfa_models
from . import tfa
from . import tfa_models
from . import utils

EPOCH_MSG = '[Epoch %d] (%dms) Posterior free-energy %.8e = KL from prior %.8e - log-likelihood %.8e'

class DeepTFA:
    """Overall container for a run of Deep TFA"""
    def __init__(self, query, mask, num_factors=tfa_models.NUM_FACTORS,
                 embedding_dim=2):
        self.num_factors = num_factors
        self.mask = mask
        self._blocks = list(query)
        for block in self._blocks:
            block.load()
            block.unload_locations()
        self.num_blocks = len(self._blocks)
        self.voxel_activations = [block.activations for block in self._blocks]
        self._blocks[-1].load()
        if tfa.CUDA:
            self.voxel_locations = self._blocks[-1].locations.pin_memory()
        else:
            self.voxel_locations = self._blocks[-1].locations
        self._templates = [block.filename for block in self._blocks]
        self._tasks = [block.task for block in self._blocks]

        self.weight_normalizers = None
        self.activation_normalizers = None
        self.normalize_activations()

        # Pull out relevant dimensions: the number of time instants and the
        # number of voxels in each timewise "slice"
        self.num_times = [acts.shape[0] for acts in self.voxel_activations]
        self.num_voxels = [acts.shape[1] for acts in self.voxel_activations]

        subjects = list(set([b.subject for b in self._blocks]))
        tasks = list(set([b.task for b in self._blocks]))
        block_subjects = [subjects.index(b.subject) for b in self._blocks]
        block_tasks = [tasks.index(b.task) for b in self._blocks]

        b = max(range(self.num_blocks), key=lambda b: self.num_times[b])
        self._blocks[b].load()
        centers, widths, weights = utils.initial_hypermeans(
            self._blocks[b].activations.numpy().T, self._blocks[b].locations.numpy(),
            num_factors
        )
        hyper_means = {
            'weights': torch.Tensor(weights),
            'factor_centers': torch.Tensor(centers),
            'factor_log_widths': widths,
        }
        self._blocks[b].unload()

        self.decoder = dtfa_models.DeepTFADecoder(self.num_factors,
                                                  embedding_dim)
        self.generative = dtfa_models.DeepTFAModel(
            self.voxel_locations, block_subjects, block_tasks,
            self.num_factors, self.num_blocks, self.num_times, embedding_dim
        )
        self.variational = dtfa_models.DeepTFAGuide(self.num_factors,
                                                    block_subjects, block_tasks,
                                                    self.num_blocks,
                                                    self.num_times,
                                                    embedding_dim, hyper_means)

    def train(self, num_steps=10, learning_rate=tfa.LEARNING_RATE,
              log_level=logging.WARNING, num_particles=tfa_models.NUM_PARTICLES,
              batch_size=64, use_cuda=True, checkpoint_steps=None,
              blocks_batch_size=4, patience=10):
        """Optimize the variational guide to reflect the data for `num_steps`"""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=log_level)
        # S x T x V -> T x S x V
        activations_loader = torch.utils.data.DataLoader(
            utils.TFADataset(self.voxel_activations),
            batch_size=batch_size,
            pin_memory=True,
        )
        decoder = self.decoder
        variational = self.variational
        generative = self.generative
        if tfa.CUDA and use_cuda:
            decoder.cuda()
            variational.cuda()
            generative.cuda()
            cuda_locations = self.voxel_locations.cuda()
        optimizer = torch.optim.Adam(list(variational.parameters()),
                                     lr=learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, min_lr=1e-5, patience=patience,
            verbose=True
        )
        variational.train()
        generative.train()

        free_energies = list(range(num_steps))
        rv_occurrences = collections.defaultdict(int)
        measure_occurrences = True

        for epoch in range(num_steps):
            start = time.time()
            epoch_free_energies = list(range(len(activations_loader)))
            epoch_lls = list(range(len(activations_loader)))
            epoch_prior_kls = list(range(len(activations_loader)))

            for (batch, data) in enumerate(activations_loader):
                epoch_free_energies[batch] = 0.0
                epoch_lls[batch] = 0.0
                epoch_prior_kls[batch] = 0.0
                block_batches = utils.chunks(list(range(self.num_blocks)),
                                             n=blocks_batch_size)
                for block_batch in block_batches:
                    if tfa.CUDA and use_cuda:
                        data = data.cuda()
                        for b in block_batch:
                            generative.module.likelihoods[b].voxel_locations =\
                                cuda_locations
                    activations = [{'Y': Variable(data[:, b, :])}
                                   for b in block_batch]
                    trs = (batch * batch_size, None)
                    trs = (trs[0], trs[0] + activations[0]['Y'].shape[0])

                    optimizer.zero_grad()
                    q = probtorch.Trace()
                    variational(decoder, q, times=trs, blocks=block_batch,
                                num_particles=num_particles)
                    p = probtorch.Trace()
                    generative(decoder, p, times=trs, guide=q,
                               observations=activations, blocks=block_batch)

                    def block_rv_weight(node, prior=True):
                        result = 1.0
                        if measure_occurrences:
                            rv_occurrences[node] += 1
                        result /= rv_occurrences[node]
                        return result
                    free_energy, ll, prior_kl = tfa.hierarchical_free_energy(
                        q, p,
                        rv_weight=block_rv_weight,
                        num_particles=num_particles
                    )

                    free_energy.backward()
                    optimizer.step()
                    epoch_free_energies[batch] += free_energy
                    epoch_lls[batch] += ll
                    epoch_prior_kls[batch] += prior_kl

                    if tfa.CUDA and use_cuda:
                        del activations
                        for b in block_batch:
                            generative.module.likelihoods[b].voxel_locations =\
                                self.voxel_locations
                        torch.cuda.empty_cache()
                if tfa.CUDA and use_cuda:
                    epoch_free_energies[batch] = epoch_free_energies[batch].cpu().data.numpy()
                    epoch_lls[batch] = epoch_lls[batch].cpu().data.numpy()
                    epoch_prior_kls[batch] = epoch_prior_kls[batch].cpu().data.numpy()
                else:
                    epoch_free_energies[batch] = epoch_free_energies[batch].data.numpy()
                    epoch_lls[batch] = epoch_lls[batch].data.numpy()
                    epoch_prior_kls[batch] = epoch_prior_kls[batch].data.numpy()

            free_energies[epoch] = np.array(epoch_free_energies).mean(0)
            scheduler.step(free_energies[epoch])

            measure_occurrences = False

            end = time.time()
            msg = EPOCH_MSG % (epoch + 1, (end - start) * 1000,
                               free_energies[epoch], np.mean(epoch_prior_kls),
                               np.mean(epoch_lls))
            logging.info(msg)
            if checkpoint_steps is not None and epoch % checkpoint_steps == 0:
                now = datetime.datetime.now()
                checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
                logging.info('Saving checkpoint...')
                self.save_state(path='.', tag=checkpoint_name)

        if tfa.CUDA and use_cuda:
            decoder.cpu()
            variational.cpu()
            generative.cpu()

        now = datetime.datetime.now()
        checkpoint_name = now.strftime(tfa.CHECKPOINT_TAG)
        logging.info('Saving checkpoint...')
        self.save_state(path='.', tag=checkpoint_name)

        return np.vstack([free_energies])

    def results(self, block=None, subject=None, task=None, hist_weights=False):
        hyperparams = self.variational.hyperparams.state_vardict()
        for k, v in hyperparams.items():
            hyperparams[k] = v.expand(1, *v.shape)

        guide = probtorch.Trace()
        if block is not None:
            subject = self.generative.block_subjects[block]
            task = self.generative.block_tasks[block]
            times = (0, self.num_times[block])
            blocks = [block]
        else:
            times = (0, max(self.num_times))
            blocks = []

        if subject is not None:
            guide.variable(torch.distributions.Normal,
                           hyperparams['subject']['mu'][:, subject],
                           softplus(hyperparams['subject']['sigma'][:, subject]),
                           value=hyperparams['subject']['mu'][:, subject],
                           name='z^P_%d' % subject)
        if task is not None:
            guide.variable(torch.distributions.Normal,
                           hyperparams['task']['mu'][:, task],
                           softplus(hyperparams['task']['sigma'][:, task]),
                           value=hyperparams['task']['mu'][:, task],
                           name='z^S_%d' % task)

        weights, factor_centers, factor_log_widths =\
            self.decoder(probtorch.Trace(), blocks,
                         self.generative.block_subjects,
                         self.generative.block_tasks, hyperparams, times,
                         guide=guide, num_particles=1)

        if block is not None:
            weights = weights[0]
            factor_centers = factor_centers[0]
            factor_log_widths = factor_log_widths[0]
        weights = weights.squeeze(0)
        factor_centers = factor_centers.squeeze(0)
        factor_log_widths = factor_log_widths.squeeze(0)

        if hist_weights:
            plt.hist(weights.view(weights.numel()).data.numpy())
            plt.show()

        result = {
            'weights': weights.data,
            'factors': tfa_models.radial_basis(self.voxel_locations,
                                               factor_centers.data,
                                               factor_log_widths.data),
            'factor_centers': factor_centers.data,
            'factor_log_widths': factor_log_widths.data,
        }
        if subject is not None:
            result['z^P_%d' % subject] = hyperparams['subject']['mu'][:, subject]
        if task is not None:
            result['z^S_%d' % task] = hyperparams['task']['mu'][:, task]
        return result

    def normalize_activations(self):
        subject_runs = list(set([(block.subject, block.run)
                                 for block in self._blocks]))
        subject_run_normalizers = {sr: 0 for sr in subject_runs}

        for block in range(len(self._blocks)):
            sr = (self._blocks[block].subject, self._blocks[block].run)
            subject_run_normalizers[sr] = max(
                subject_run_normalizers[sr],
                torch.abs(self.voxel_activations[block]).max()
            )

        self.activation_normalizers =\
            [subject_run_normalizers[(block.subject, block.run)]
             for block in self._blocks]
        return self.activation_normalizers

    def plot_factor_centers(self, block, filename=None, show=True, t=None,
                            labeler=None):
        if labeler is None:
            labeler = lambda b: b.task
        results = self.results(block)

        centers_sizes = np.repeat([50], self.num_factors)
        sizes = torch.exp(results['factor_log_widths']).numpy()

        centers = results['factor_centers'].numpy()

        plot = niplot.plot_connectome(
            np.eye(self.num_factors * 2),
            np.vstack([centers, centers]),
            node_size=np.vstack([sizes, centers_sizes]),
            title="Block %d (Participant %d, Run %d, Stimulus: %s)" %\
                  (block, self._blocks[block].subject, self._blocks[block].run,
                   labeler(self._blocks[block]))
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_original_brain(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0, labeler=None, **kwargs):
        if labeler is None:
            labeler = lambda b: b.task
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]
        if self.activation_normalizers is None:
            self.normalize_activations()

        image = utils.cmu2nii(self.voxel_activations[block].numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[block])
        image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title="Block %d (Participant %d, Run %d, Stimulus: %s)" %\
                  (block, self._blocks[block].subject, self._blocks[block].run,
                   labeler(self._blocks[block])),
            vmin=-self.activation_normalizers[block],
            vmax=self.activation_normalizers[block],
            **kwargs,
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_reconstruction(self, block=None, filename=None, show=True,
                            plot_abs=False, t=0, labeler=None, **kwargs):
        if labeler is None:
            labeler = lambda b: b.task
        if block is None:
            block = np.random.choice(self.num_blocks, 1)[0]
        if self.activation_normalizers is None:
            self.normalize_activations()

        results = self.results(block)

        reconstruction = results['weights'] @ results['factors']

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[block])
        image_slice = nilearn.image.index_img(image, t)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title="Block %d (Participant %d, Run %d, Stimulus: %s)" %\
                  (block, self._blocks[block].subject, self._blocks[block].run,
                   labeler(self._blocks[block])),
            vmin=-self.activation_normalizers[block],
            vmax=self.activation_normalizers[block],
            **kwargs,
        )

        logging.info(
            'Reconstruction Error (Frobenius Norm): %.8e out of %.8e',
            np.linalg.norm(
                (self.voxel_activations[block] - reconstruction).numpy()
            ),
            np.linalg.norm(self.voxel_activations[block].numpy())
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_subject_template(self, subject, filename=None, show=True,
                              plot_abs=False, **kwargs):
        i = list(set([block.subject for block in self._blocks])).index(subject)
        results = self.results(block=None, task=None, subject=i)
        template = [i for (i, b) in enumerate(self._blocks)
                    if b.subject == subject][0]
        reconstruction = results['weights'] @ results['factors']

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[template])
        image_slice = nilearn.image.index_img(image, 0)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title="Template for Participant %d" % subject,
            vmin=-self.activation_normalizers[template],
            vmax=self.activation_normalizers[template],
            **kwargs,
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def plot_task_template(self, task, filename=None, show=True, plot_abs=False,
                           labeler=lambda x: x, **kwargs):
        i = self._tasks.index(task)
        results = self.results(block=None, subject=None, task=i)
        template = [i for (i, b) in enumerate(self._blocks)
                    if b.task == task][0]
        reconstruction = results['weights'] @ results['factors']

        image = utils.cmu2nii(reconstruction.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[template])
        image_slice = nilearn.image.index_img(image, 0)
        plot = niplot.plot_glass_brain(
            image_slice, plot_abs=plot_abs, colorbar=True, symmetric_cbar=True,
            title="Template for Stimulus '%s'" % labeler(task),
            vmin=-self.activation_normalizers[template],
            vmax=self.activation_normalizers[template],
            **kwargs,
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        return plot

    def visualize_factor_embedding(self, filename=None, show=True,
                                   hist_log_widths=True, **kwargs):
        results = self.results(block=None, subject=0, task=None)
        centers = results['factor_centers']
        log_widths = results['factor_log_widths']
        widths = torch.exp(log_widths)

        plot = niplot.plot_connectome(
            np.eye(self.num_factors),
            centers.view(self.num_factors, 3).numpy(),
            node_size=widths.view(self.num_factors).numpy(),
            title="$x^F$ std-dev %.8e, $\\rho^F$ std-dev %.8e" %
            (centers.std(0).norm(), log_widths.std(0).norm()),
            **kwargs
        )

        if filename is not None:
            plot.savefig(filename)
        if show:
            niplot.show()

        if hist_log_widths:
            plt.hist(log_widths.view(log_widths.numel()).numpy())
            plt.show()

        return plot, centers, log_widths

    def scatter_subject_embedding(self, labeler=None, filename=None, show=True,
                                  xlims=None, ylims=None, figsize=(3.75, 2.75),
                                  colormap='Set1'):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_p_mu = hyperparams['subject']['mu'].data
        z_p_sigma = softplus(hyperparams['subject']['sigma'].data)

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = np.unique([l for l in labels if l is not None])
        palette = dict(zip(all_labels,
                           utils.compose_palette(len(all_labels),
                                                 colormap=colormap)))

        subjects = list(set([block.subject for block in self._blocks]))
        z_ps = torch.stack(
            [torch.normal(z_p_mu[subjects.index(b.subject)],
                          z_p_sigma[subjects.index(b.subject)])
             for b in self._blocks if labeler(b) is not None]
        )
        block_colors = [palette[labeler(b)] for b in self._blocks
                        if labeler(b) is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^P_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^P_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Participant Embeddings')
        ax.scatter(x=z_ps[:, 0], y=z_ps[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            fig.show()

    def scatter_task_embedding(self, labeler=None, filename=None, show=True,
                               xlims=None, ylims=None, figsize=(3.75, 2.75),
                               colormap='Set1'):
        hyperparams = self.variational.hyperparams.state_vardict()
        z_s_mu = hyperparams['task']['mu'].data
        z_s_sigma = softplus(hyperparams['task']['sigma'].data)

        if labeler is None:
            labeler = lambda b: b.default_label()
        labels = [labeler(b) for b in self._blocks]
        all_labels = np.unique([l for l in labels if l is not None])
        palette = dict(zip(all_labels,
                           utils.compose_palette(len(all_labels),
                                                 colormap=colormap)))

        tasks = list(set([block.task for block in self._blocks]))
        z_ss = torch.stack(
            [torch.normal(z_s_mu[tasks.index(b.task)],
                          z_s_sigma[tasks.index(b.task)])
             for b in self._blocks if labeler(b) is not None]
        )
        block_colors = [palette[labeler(b)] for b in self._blocks
                        if labeler(b) is not None]

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(111, facecolor='white')
        fig.axes[0].set_xlabel('$z^S_1$')
        if xlims is not None:
            fig.axes[0].set_xlim(*xlims)
        fig.axes[0].set_ylabel('$z^S_2$')
        if ylims is not None:
            fig.axes[0].set_ylim(*ylims)
        fig.axes[0].set_title('Stimulus Embeddings')
        ax.scatter(x=z_ss[:, 0], y=z_ss[:, 1], c=block_colors)
        utils.palette_legend(list(palette.keys()), list(palette.values()))

        if filename is not None:
            fig.savefig(filename)
        if show:
            fig.show()

    def common_name(self):
        return os.path.commonprefix([os.path.basename(b.filename)
                                     for b in self._blocks])

    def save_state(self, path='.', tag=''):
        name = self.common_name() + tag
        variational_state = self.variational.state_dict()
        torch.save(variational_state,
                   path + '/' + name + '.dtfa_guide')
        torch.save(self.generative.state_dict(),
                   path + '/' + name + '.dtfa_model')

    def save(self, path='.'):
        name = self.common_name()
        torch.save(self.variational.state_dict(),
                   path + '/' + name + '.dtfa_guide')
        torch.save(self.generative.state_dict(),
                   path + '/' + name + '.dtfa_model')
        with open(path + '/' + name + '.dtfa', 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def load_state(self, basename):
        model_state = torch.load(basename + '.dtfa_model')
        self.generative.load_state_dict(model_state)

        guide_state = torch.load(basename + '.dtfa_guide')
        self.variational.load_state_dict(guide_state)

    @classmethod
    def load(cls, basename):
        with open(basename + '.dtfa', 'rb') as pickle_file:
            dtfa = pickle.load(pickle_file)
        dtfa.load_state(basename)

        return dtfa

    def decoding_accuracy(self, labeler=lambda x: x, window_size=60):
        """
        :return: accuracy: a dict containing decoding accuracies for each task [activity,isfc,mixed]
        """
        tasks = np.unique([labeler(b.task) for b in self._blocks])
        group = {task: [] for task in tasks}
        accuracy = {task: {'node': [], 'isfc': [], 'mixed': [], 'kl': []}
                    for task in tasks}

        for (b, block) in enumerate(self._blocks):
            factorization = self.results(b)
            group[(block.task)].append(factorization['weights'])

        for task in set(tasks):
            print(task)
            group[task] = torch.stack(group[task])
            if group[task].shape[0] < 2:
                raise ValueError('Not enough subjects for task %s' % task)
            group1 = group[task][:group[task].shape[0] // 2]
            group2 = group[task][group[task].shape[0] // 2:]
            node_accuracy, node_correlation = utils.get_decoding_accuracy(
                group1.data.numpy(), group2.data.numpy(), window_size
            )
            accuracy[task]['node'].append(node_accuracy)
            isfc_accuracy, isfc_correlation = utils.get_isfc_decoding_accuracy(
                group1.data.numpy(), group2.data.numpy(), window_size
            )
            accuracy[task]['isfc'].append(isfc_accuracy)
            accuracy[task]['mixed'].append(
                utils.get_mixed_decoding_accuracy(node_correlation,
                                                  isfc_correlation)
            )
            accuracy[task]['kl'].append(
                utils.get_kl_decoding_accuracy(group1.data.numpy(),
                                               group2.data.numpy(), window_size)
            )

        return accuracy

    def voxel_decoding_accuracy(self, labeler=lambda x: x, window_size=60):
        times = self.num_times
        keys = np.unique([labeler(b.task) for b in self._blocks])
        group = {key: [] for key in keys}
        accuracy = {key: [] for key in keys}
        for key in keys:
            print(key)
            for n in range(self.num_blocks):
                if key == self._blocks[n].task:
                    self._blocks[n].load()
                    group[key].append(self._blocks[n].activations[:times[n], :])
            group[key] = np.rollaxis(np.dstack(group[key]), -1)
            if group[key].shape[0] < 2:
                raise ValueError('not enough subjects for the task: ' + key)
            else:
                G1 = group[key][:int(group[key].shape[0] / 2), :, :]
                G2 = group[key][int(group[key].shape[0] / 2):, :, :]
                accuracy[key].append(
                    utils.get_decoding_accuracy(G1, G2, window_size)[0]
                )
        return accuracy
