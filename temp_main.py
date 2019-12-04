import logging
import numpy as np
import htfa_torch.dtfa as DTFA
import htfa_torch.niidb as niidb
import htfa_torch.utils as utils
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
import os
from torch.nn.functional import softplus
import torch
import itertools
from htfa_torch import tfa_models
import nilearn.plotting as niplot
import imageio

def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1), np.linspace(p1[1], p2[1], parts+1))

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

affvids_db = niidb.FMriActivationsDb('data/affvids2018.db')

SUBJECT_IDs = [4,11,12,9]
new_db = [b for b in affvids_db.all() if 'rest' not in b.task and 'irrelevant' not in b.task and b.subject in SUBJECT_IDs]
subject_list = np.unique([b.subject for b in new_db])
use_db = []

for subject in subject_list:
    print (subject)
    temp_db = [b for b in new_db if b.subject == subject]
    process_fears = ['spider','social','heights']
    for process_fear in process_fears:
        ratings = []
        block_indices = []
        for i in range(len(temp_db)):
            if process_fear in temp_db[i].task:
                block_indices.append(i)
                ratings.append([temp_db[i].individual_differences['fear_rating']])
        ratings = np.array(ratings)
        sorting = np.argsort(ratings[:,0])
        print (np.sort(ratings[:,0]))
        for i in range(len(sorting)):
            if i<=5:
                temp_db[block_indices[sorting[i]]].task = process_fear + '_low'
    #         elif 3<i<=7:
    #             new_db[block_indices[sorting[i]]].task =  process_fear +'_medium'
            else:
                temp_db[block_indices[sorting[i]]].task =  process_fear +'_high'
    use_db.extend(temp_db)

dtfa = DTFA.DeepTFA(use_db, mask='/home/zulqarnain/fmri_data/AffVids/nifti/wholebrain2.nii.gz', num_factors=300, embedding_dim=2)
# dtfa.load_state('/home/zulqarnain/Documents/AlgorithmX_results/Subjects_[4,9,11,12]/sub-CHECK_11262019_125313')
dtfa.load_state('sub-CHECK_12032019_174801')


self = dtfa

hyperparams = self.variational.hyperparams.state_vardict()
tasks = self.tasks()
subjects = self.subjects()
interactions = OrderedSet(list(itertools.product(subjects, tasks)))
mean_activations = [a.mean(dim=0) for a in self.voxel_activations]
original_baseline_brain = torch.stack(mean_activations).mean(dim=0).unsqueeze(0)
vmax = torch.max(original_baseline_brain).data.numpy()

image = utils.cmu2nii(original_baseline_brain.data.numpy(),
                      self.voxel_locations.numpy(),
                      self._templates[0])
filename = 'results/original_baseline_brain.png'
plot = niplot.plot_glass_brain(
    image, plot_abs=False, colorbar=True, symmetric_cbar=True,
    title="Original Baseline Brain",
    vmin=-vmax, vmax=vmax, output_file=filename)

for s in range(len(subjects)):
    participant_blocks = [b.activations.mean(0) for b in dtfa._blocks if b.subject == subjects[s]]
    mean_activations = torch.stack(participant_blocks).mean(dim=0).unsqueeze(0)
    vmax = torch.max(mean_activations).data.numpy()
    image = utils.cmu2nii(mean_activations.data.numpy(),
                          self.voxel_locations.numpy(),
                          self._templates[0])
    filename = 'results/subject_' + str(subjects[s]) + '_original_participant_brain.png'
    plot = niplot.plot_glass_brain(
        image, plot_abs=False, colorbar=True, symmetric_cbar=True,
        title="Participant Brain, Participant %d" % subjects[s],
        vmin=-vmax, vmax=vmax, output_file=filename)

for t in range(len(tasks)):
    task_blocks = [b.activations.mean(0) for b in dtfa._blocks if b.task == tasks[t]]
    mean_activations = torch.stack(task_blocks).mean(dim=0).unsqueeze(0)
    vmax = torch.max(mean_activations).data.numpy()
    image = utils.cmu2nii(mean_activations.data.numpy(),
                          self.voxel_locations.numpy(),
                          self._templates[0])
    filename = 'results/task_' + str(tasks[t]) + '_original_task_brain.png'
    plot = niplot.plot_glass_brain(
        image, plot_abs=False, colorbar=True, symmetric_cbar=True,
        title="Task Brain, Task %s" % tasks[t],
        vmin=-vmax, vmax=vmax, output_file=filename)

for s in range(len(subjects)):
    participant_blocks = [b.activations.mean(0) for b in dtfa._blocks if b.subject == subjects[s]]
    participant_activations = torch.stack(participant_blocks).mean(dim=0).unsqueeze(0)
    for t in range(len(tasks)):
        task_blocks = [b.activations.mean(0) for b in dtfa._blocks if b.task == tasks[t]]
        task_activations = torch.stack(task_blocks).mean(dim=0).unsqueeze(0)
        interaction_blocks = [b.activations.mean(0) for b in dtfa._blocks if (b.subject == subjects[s] and b.task == tasks[t])]
        mean_activations = torch.stack(interaction_blocks).mean(dim=0).unsqueeze(0)
        vmax = torch.max(mean_activations).data.numpy()
        image = utils.cmu2nii(mean_activations.data.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[0])
        filename = 'results/subject_task_' + str(subjects[s]) + '_' + str(tasks[t]) + '_original_interaction_brain.png'
        plot = niplot.plot_glass_brain(
            image, plot_abs=False, colorbar=True, symmetric_cbar=True,
            title="Participant Brain, Participant %d Stimulus %s" % (subjects[s],tasks[t]),
            vmin=-vmax, vmax=vmax, output_file=filename)
        difference_brain = mean_activations - (participant_activations + task_activations - original_baseline_brain)
        vmax = torch.max(difference_brain).data.numpy()
        image = utils.cmu2nii(difference_brain.data.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[0])
        filename = 'results/subject_task_' + str(subjects[s]) + '_' + str(tasks[t]) + '_original_difference_brain.png'
        plot = niplot.plot_glass_brain(
            image, plot_abs=False, colorbar=True, symmetric_cbar=True,
            title="Participant Brain, Participant %d Stimulus %s" % (subjects[s],tasks[t]),
            vmin=-vmax, vmax=vmax, output_file=filename)

z_pw_mu = hyperparams['participant_weight']['mu'].data
z_sw_mu = hyperparams['stimulus_weight']['mu'].data
z_p_mu = hyperparams['subject']['mu'].data
for s in range(len(subjects)):
    print (subjects[s])
    W_map = hyperparams['weights']['mu'].data
    # W_mean_universal = W_map.mean(0).mean(0)
    W_mean_universal = hyperparams['global_weight_mean']['mu'].data
    W_mean_participant = self.decoder.participant_weights_embedding(z_pw_mu[s,:]).view(-1,
                                                                  self.num_factors)
    #idx = np.where(np.asarray(dtfa.variational.block_subjects) == s)[0]
    # W_mean_participant = hyperparams['participant_weight_mean']['mu'].data
    # W_mean_participant = W_mean_participant[s,:]
    factor_params = self.decoder.factors_embedding(z_p_mu[s,:]).view(
        -1, self.num_factors, 4, 2
    )
    centers_predictions = factor_params[:, :, :3, 0]
    log_widths_predictions = factor_params[:, :, 3, 0]
    mean_factors = tfa_models.radial_basis(self.voxel_locations,
                                           centers_predictions.data,
                                           log_widths_predictions.data)[0, :, :]

    F_mu_map = hyperparams['factor_centers']['mu'].data
    F_width_map = hyperparams['factor_log_widths']['mu'].data
    F_participant = tfa_models.radial_basis(self.voxel_locations,
                                                               F_mu_map[s,:,:],
                                                               F_width_map[s,:])
    participant_brain = W_mean_participant @ F_participant
    # participant_brain = participant_brain.unsqueeze(0)
    vmax = torch.max(participant_brain).data.numpy()
    image = utils.cmu2nii(participant_brain.data.numpy(),
                          self.voxel_locations.numpy(),
                          self._templates[0])
    filename = 'results/subject_' + str(subjects[s]) + '_participant_brain.png'
    plot = niplot.plot_glass_brain(
        image, plot_abs=False, colorbar=True, symmetric_cbar=True,
        title="Participant Brain, Participant %d" % subjects[s],
        vmin=-vmax, vmax=vmax, output_file=filename)

    baseline_brain = W_mean_universal @ F_participant
    vmax = torch.max(baseline_brain).data.numpy()
    # baseline_brain = baseline_brain.unsqueeze(0)
    image = utils.cmu2nii(baseline_brain.data.numpy(),
                          self.voxel_locations.numpy(),
                          self._templates[130])
    filename = 'results/subject_' + str(subjects[s]) + '_baseline_brain.png'
    plot = niplot.plot_glass_brain(
        image, plot_abs=False, colorbar=True, symmetric_cbar=True,
        title="Baseline Brain, Participant %d" % subjects[s],
        vmin=-vmax, vmax=vmax, output_file=filename)


    for t in range(len(tasks)):
        # idx = np.where(np.asarray(dtfa.variational.block_tasks) == t)[0]
        W_mean_stimulus = self.decoder.stimulus_weights_embedding(z_sw_mu[t,:]).view(-1,
                                                                  self.num_factors)
        # W_mean_stimulus = hyperparams['stimulus_weight_mean']['mu'].data
        # W_mean_stimulus = W_mean_stimulus[t,:]

        task_brain = W_mean_stimulus @ F_participant
        vmax = torch.max(task_brain).data.numpy()
        # task_brain = task_brain.unsqueeze(0)
        image = utils.cmu2nii(task_brain.data.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[0])
        filename = 'results/subject_' + str(subjects[s]) + '_task_' + tasks[t] + 'mean_brain.png'
        plot = niplot.plot_glass_brain(
            image, plot_abs=False, colorbar=True, symmetric_cbar=True,
            title="Task Brain, Participant %d Task %s" % (subjects[s],tasks[t]),
            vmin=-vmax, vmax=vmax, output_file=filename)
        idx = np.where((np.asarray(dtfa.variational.block_tasks) == t) & (np.asarray(dtfa.variational.block_subjects) == s))[0]
        W_mean_interaction = W_map[idx,:,:].mean(0).mean(0)
        interaction_brain = W_mean_interaction @ F_participant
        interaction_brain = interaction_brain.unsqueeze(0)
        vmax = torch.max(interaction_brain).data.numpy()
        image = utils.cmu2nii(interaction_brain.data.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[0])
        filename = 'results/subject_' + str(subjects[s]) + '_task_' + tasks[t] + '_interaction_brain.png'
        plot = niplot.plot_glass_brain(
            image, plot_abs=False, colorbar=True, symmetric_cbar=True,
            title="Task Brain, Participant %d Task %s interaction" % (subjects[s],tasks[t]),
            vmin=-vmax, vmax=vmax, output_file=filename)

        linear_interaction_brain = (W_mean_participant + W_mean_stimulus - W_mean_universal) @ F_participant
        # linear_interaction_brain = linear_interaction_brain.unsqueeze(0)
        vmax = torch.max(linear_interaction_brain).data.numpy()
        image = utils.cmu2nii(linear_interaction_brain.data.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[0])
        filename = 'results/subject_' + str(subjects[s]) + '_task_' + tasks[t] + '_interaction_linear_brain.png'
        plot = niplot.plot_glass_brain(
            image, plot_abs=False, colorbar=True, symmetric_cbar=True,
            title="Task Brain, Participant %d Task %s interaction,linear" % (subjects[s], tasks[t]),
            vmin=-vmax, vmax=vmax, output_file=filename)

        # difference_interaction_brain = (W_mean_interaction -
        #                                 (W_mean_participant + W_mean_stimulus - W_mean_universal)) @ F_participant
        difference_interaction_brain = interaction_brain - (participant_brain + task_brain - baseline_brain)
        # difference_interaction_brain = difference_interaction_brain.unsqueeze(0)
        vmax = torch.max(difference_interaction_brain).data.numpy()
        image = utils.cmu2nii(difference_interaction_brain.data.numpy(),
                              self.voxel_locations.numpy(),
                              self._templates[0])
        filename = 'results/subject_' + str(subjects[s]) + '_task_' + tasks[t] + '_interaction_difference_brain.png'
        plot = niplot.plot_glass_brain(
            image, plot_abs=False, colorbar=True, symmetric_cbar=True,
            title="Task Brain, Participant %d Task %s interaction,diff" % (subjects[s], tasks[t]),
            vmin=-vmax, vmax=vmax, output_file=filename)


z_ps_mu = hyperparams['interactions']['mu'].data

z_p_mu = hyperparams['subject']['mu'].data
for use_subject in subject_list:
    print (use_subject)
    for stimulus in ['spider','social','heights']:
        # temp_z_ps = z_ps_mu[4,:]
        idx = [id for (id,inter) in enumerate(interactions) if inter == (use_subject,stimulus+'_high')][0]
        print(idx)
        temp_z_ps_high = z_ps_mu[idx]
        idx = [id for (id,inter) in enumerate(interactions) if inter == (use_subject,stimulus+'_low')][0]
        temp_z_ps_low = z_ps_mu[idx]
        use_points = list(getEquidistantPoints(temp_z_ps_low.data,temp_z_ps_high.data,10))
        temp_z_p = z_p_mu[np.where(np.asarray(subjects) == use_subject)[0][0],:]
        filenames = []
        for (i,points) in enumerate(use_points):
            weight_predictions = self.decoder.weights_embedding(torch.Tensor(np.array(points))).view(
                -1, self.num_factors, 2
            )
            mean_weight = weight_predictions[:,:,0]

            factor_params = self.decoder.factors_embedding(temp_z_p).view(
                -1, self.num_factors, 4, 2
            )
            centers_predictions = factor_params[:, :, :3,0]
            log_widths_predictions = factor_params[:, :, 3,0]
            mean_factors = tfa_models.radial_basis(self.voxel_locations,
                                                           centers_predictions.data,
                                                           log_widths_predictions.data)[0,:,:]
            mean_brain = mean_weight @ mean_factors
            vmax = torch.max(mean_brain).data.numpy()
            image = utils.cmu2nii(mean_brain.data.numpy(),
                                  self.voxel_locations.numpy(),
                                  self._templates[0])
            filenames.append('results/gifs/subject_' + str(use_subject) + '_stimulus_' + str(stimulus) + '_' +str(i) + '.png')
            plot = niplot.plot_glass_brain(
                image, plot_abs=False, colorbar=True, symmetric_cbar=True,
                title="Mean Image of Interaction %d" % i,
                vmin=-vmax, vmax=vmax,output_file=filenames[-1])


        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('results/gifs/subject_' + str(use_subject) + '_stimulus_' + stimulus + '.gif', images)
r = 3