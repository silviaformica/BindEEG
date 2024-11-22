# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:30:12 2021

@author: Silvia Formica
"""

import mne
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as op

base_path = 'E:/BindEEG/'
os.chdir(base_path)


###############################################################################################################
## Importing anatomy template and specific functions
###############################################################################################################

from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs
from mne.time_frequency import tfr_morlet
from mne.minimum_norm import source_induced_power, source_band_induced_power

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

# src = mne.setup_source_space('fsaverage', 'oct6', subjects_dir=subjects_dir)
# src.save('C:/Users/silvi/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-oct-6-src.fif')

src_name = op.join(fs_dir, 'bem', 'fsaverage-oct-6-src.fif')
src = mne.read_source_spaces(src_name, verbose = True)

conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject='fsaverage',
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

# mne.viz.plot_bem(subject='fsaverage', subjects_dir=subjects_dir,
#                  brain_surfaces='white', src=src, orientation='coronal')


# fig = mne.viz.plot_alignment(subject='fsaverage', subjects_dir=subjects_dir,
#                              surfaces='white', coord_frame='head',
#                              src=src)
# mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
#                     distance=0.30, focalpoint=(-0.03, -0.01, 0.03))


###############################################################################################################

###############################################################################################################

subjlist = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

for idx, val in enumerate(subjlist):
    
    print('-----------------------------------\n-----------------------------------')
    print('Processing partitipant ', str(val))
    print('-----------------------------------\n-----------------------------------')
    
    
    data_path = base_path + 'ParticipantsData/Subj' + str(val) + '/'
    os.chdir(data_path)
    subj = mne.read_epochs('Subj' + str(val) + '-epo.fif')
    
    # keeping only correct trials
    subj = subj[subj.metadata['acc'] == 1].crop(-0.5, 2.5)
    
    ####################################################################
    # Computing subject-specific preliminaries for source reconstruction
    ####################################################################
    
    # Check that the locations of EEG electrodes is correct with respect to MRI
    # mne.viz.plot_alignment(
    # subj.info, src=src, eeg=['original','projected'], trans=trans,
    # show_axes=True, mri_fiducials=True, dig='fiducials')   
    

    fwd = mne.make_forward_solution(subj.info, trans=trans, src=src,
                            bem=bem, eeg=True, mindist=5.0, n_jobs = 4, verbose=False)
    #print(fwd)


    noise_cov = mne.compute_covariance(
    subj, tmin = -0.5, tmax=-0.2, method=['shrunk', 'empirical'], rank=None, verbose=False)

    # fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, subj.info)
        
    inv = make_inverse_operator(
        subj.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)



    ####################################################################
    # Compute sources
    ####################################################################

    bands = dict(theta=[3,7], alpha=[8, 14], beta=[15, 30], gamma = [30, 40])
    n_cycles = np.array([3., 3., 3., 3., 3.,
                         4., 4., 4., 4., 4., 4., 4.,
                         5., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
                         5., 5., 5., 5., 5., 5.,
                         6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.])


    # Implementation, LL
    
    this_power = source_band_induced_power(
    subj['I_LL'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Impl_LL_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Impl_LL_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Impl_LL_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Impl_LL_%s' % val)
    
    # Implementation, LR
    
    this_power = source_band_induced_power(
    subj['I_LR'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Impl_LR_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Impl_LR_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Impl_LR_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Impl_LR_%s' % val)
    
    # Implementation, RL
    
    this_power = source_band_induced_power(
    subj['I_RL'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Impl_RL_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Impl_RL_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Impl_RL_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Impl_RL_%s' % val)
    
    # Implementation, RR
    
    this_power = source_band_induced_power(
    subj['I_RR'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Impl_RR_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Impl_RR_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Impl_RR_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Impl_RR_%s' % val)
    
    # Memorization, LL
    
    this_power = source_band_induced_power(
    subj['M_LL'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Memo_LL_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Memo_LL_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Memo_LL_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Memo_LL_%s' % val)
    
    # Memorization, LR
    
    this_power = source_band_induced_power(
    subj['M_LR'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Memo_LR_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Memo_LR_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Memo_LR_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Memo_LR_%s' % val)
    
    # Memorization, RL
    
    this_power = source_band_induced_power(
    subj['M_RL'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Memo_RL_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Memo_RL_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Memo_RL_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Memo_RL_%s' % val)
    
    # Memorization, RR
    
    this_power = source_band_induced_power(
    subj['M_RR'], 
    inv, bands, method = 'dSPM', n_cycles = n_cycles, n_jobs = 4, use_fft = False, decim = 4)

    os.chdir(base_path + '/Sources/theta_whole_brain')
    this_power['theta'].save('Memo_RR_%s' % val)
    os.chdir(base_path + '/Sources/alpha_whole_brain')
    this_power['alpha'].save('Memo_RR_%s' % val)
    os.chdir(base_path + '/Sources/beta_whole_brain')
    this_power['beta'].save('Memo_RR_%s' % val)
    os.chdir(base_path + '/Sources/gamma_whole_brain')
    this_power['gamma'].save('Memo_RR_%s' % val)
    
    
    