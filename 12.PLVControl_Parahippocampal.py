# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:12:21 2021

@author: Silvia
"""

# CONTROL ANALYSES FOR PLV
# Connectivity between mPFC and a control region: PARAHIPPOCAMPAL ROI



import mne
import numpy as np 
import pandas as pd
import os
import os.path as op
from scipy import signal, stats, spatial
import pickle

base_path = 'D:/BindEEG/'
os.chdir(base_path)

# function to find index of desired timepoint in array of timepoints
def find_nearest_tp(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

###############################################################################################################
## Importing anatomy template and specific functions
###############################################################################################################

from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

# src = mne.setup_source_space('fsaverage', 'oct6', subjects_dir=subjects_dir)
# src.save('C:/Users/Silvia/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-oct-6-src.fif')

src_name = op.join(fs_dir, 'bem', 'fsaverage-oct-6-src.fif')
src = mne.read_source_spaces(src_name, verbose = True)

# conductivity = (0.3, 0.006, 0.3)  # for three layers
# model = mne.make_bem_model(subject='fsaverage',
#                             conductivity=conductivity,
#                             subjects_dir=subjects_dir)
# bem = mne.make_bem_solution(model)

# bem_name = op.join(fs_dir, 'bem', 'fsaverage-oct-6-bem.fif')
# mne.write_bem_solution(bem_name, bem)

bem_name = op.join(fs_dir, 'bem', 'fsaverage-oct-6-bem.fif')
bem = mne.read_bem_solution(bem_name, verbose = True)

###############################################################################################################
## ROIs
###############################################################################################################

labels = mne.read_labels_from_annot(
'fsaverage', 'aparc', 'both', subjects_dir=subjects_dir)   

# Getting rid of the empty label
labels = labels[:-1]

label_L = [label for label in labels if label.name == 'caudalanteriorcingulate-lh'][0]
label_R = [label for label in labels if label.name == 'caudalanteriorcingulate-rh'][0]

mPFC = mne.BiHemiLabel(label_L, label_R)

## Control

control_lh = [label for label in labels if label.name == 'parahippocampal-lh'][0]
control_rh = [label for label in labels if label.name == 'parahippocampal-rh'][0]



labels = list()
labels.append(mPFC)
labels.append(control_lh)
labels.append(control_rh)

###############################################################################################################
## Computing Seed to all connectivity - trial resolved
###############################################################################################################

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

#conditions = ['I_LL', 'I_LR', 'I_RL', 'I_RR', 'M_LL', 'M_LR', 'M_RL', 'M_RR']

subjlist = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

## START LOOPING ACROSS SUBJECTS

for idx_s, val in enumerate(subjlist):
    
    print('-----------------------------------\n-----------------------------------')
    print('Processing partitipant ', str(val))
    print('-----------------------------------\n-----------------------------------')

    data_path = base_path + 'ParticipantsData/Subj' + str(val) + '/'
    os.chdir(data_path)
    subj = mne.read_epochs('Subj' + str(val) + '-epo.fif', verbose = False)
    
    # keeping only correct trials
    subj = subj.pick_types(eeg = True)
    
    ####################################################################
    # Computing subject-specific preliminaries for source reconstruction
    ####################################################################
    
    # # check_visualthat the locations of EEG electrodes is correct with respect to MRI
    # mne.viz.plot_alignment(
    # subj.info, src=src, eeg=['original','projected'], trans=trans,
    # show_axes=True, mri_fiducials=True, dig='fiducials')   
    
    print('--- Forward Model ---')
    fwd = mne.make_forward_solution(subj.info, trans=trans, src=src,
                            bem=bem, eeg=True, mindist=5.0, n_jobs = 4, verbose = False)
    #print(fwd)
    
    print('--- Noise Covariance ---')    
    noise_cov = mne.compute_covariance(
    subj, tmin = -0.5, tmax=-0.2, method=['shrunk', 'empirical'], rank=None, verbose=False)
    
    # fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, subj.info)
        
    print('--- Inverse Operator ---')
    inv = make_inverse_operator(
        subj.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose = False)

    #filtering in theta range and downsampling to 128 Hz
    subj_filt = subj.copy().filter(l_freq = 3, h_freq = 7, 
                method = 'fir').decimate(4)
    # Timepoints at which the signal will be cut later after hilbert
    val0 = 0.54  # beginning of significant cluster
    val18 = 0.8  # end of significant cluster
    valmid = 0.67
    # I take a time window of 633 ms around the center of the cluster
    valstart = 0.67 - (0.63/2)
    valend =  0.67 + (0.63/2)
    idx0, tp1 = find_nearest_tp(subj_filt.times, valstart)
    idx18, tp3 = find_nearest_tp(subj_filt.times, valend) 

    ####################################################################
    # Connectivity across all trials
    ####################################################################

    # Projecting filtered signal on the sources
    stc_source = apply_inverse_epochs(subj_filt, inv, lambda2,
    method, pick_ori="normal", verbose = False, return_generator = False)  
           
    n_perm = 100
     
    PLV_all = np.zeros(shape = [len(stc_source), len(labels)])
    PLV_all_perm = np.zeros(shape = [len(stc_source), len(labels), n_perm])

    
    # Looping across trials and extracting values for mPFC  
    data_mPFC = np.zeros(shape = [stc_source[0].in_label(mPFC).data.shape[0], 
                                  len(stc_source[0].times), len(stc_source)])      
    for t in range(len(stc_source)):
        data_mPFC[:, :, t] = stc_source[t].in_label(mPFC).data    
    
    
    
    ## LOOPING ACROSS LABELS 
    for idx_l, label in enumerate(labels):

        print('--- LABEL:' , str(idx_l + 1), ' - ' , str(label.name), '---')
        
        data_other = np.zeros(shape=[stc_source[0].in_label(label).data.shape[0], 
                            len(stc_source[0].times), len(stc_source)])

        for t in range(len(stc_source)):
            data_other[:, :, t] = stc_source[t].in_label(label).data
                    
        ## COMPUTING CONNECTIVITY BETWEEN mPFC and CURRENT LABEL
        data = np.vstack([data_mPFC, data_other])
   
        # Applying hilbert transform        
        data_hilbert = signal.hilbert(data)
    
        # Need to crop in CTI
        data_hilbert = data_hilbert[:, idx0:idx18, :]
    
        ### Code from Bruña et al. 2018
        nc, ns, nt = data_hilbert.shape 
        ndat = np.divide(data_hilbert, abs(data_hilbert))
        
        plv = np.zeros([nc, nc, nt])
           
        for t in range(0, nt):   
            plv[:,:, t] = abs(ndat[:, :, t] @ ndat[:,:,t].conj().T) / ns
    
        # Computing RMS and store in PLV_all
        for t in range(0, plv.shape[-1]):        
            PLV_all[t, idx_l] = np.sqrt(np.mean(np.square(plv[:, :, t])))

        # ## DIAGONALLY THE TRADITIONAL ONE IS 1
        # # and also the new formulation must be!
        # # traditional plv code for comparison        
        # phs = np.angle(data_hilbert)
        # plv_trad = np.zeros([nc, nc, nt])
        # for t in range(0, 1):
        #     for c1 in range(nc):
        #         for c2 in range(nc):
        #             dphs = phs[c1, :, t] - phs[c2, :, t]
        #             plv_trad[c1, c2, t] = abs(np.mean(np.exp(1j * dphs)))    



    ####################################################################
    # NULL DISTRIBUTION
    # This part of code runs n_perm shuffling of the mPFC and the 
    # data from the label, permuting across trials.
    # This should create a null distribution of connectivity values
    # against which I could select significantly connected ROIs
    ####################################################################

    # for n_p in range(n_perm):
        
    #     print('--- n_Perm:' , str(n_p), '---')
        
    #     rng = np.random.default_rng()
    #     rng.shuffle(data_mPFC, axis = 2)
    #     rng.shuffle(data_other, axis = 2)
            
    #     ## COMPUTING CONNECTIVITY BETWEEN mPFC and CURRENT LABEL
    #     data = np.vstack([data_mPFC, data_other])
   
    #     # Applying hilbert transform        
    #     data_hilbert = signal.hilbert(data)
    
    #     # Need to crop in CTI
    #     data_hilbert = data_hilbert[:, idx0:idx18, :]
    
    #     ### Code from Bruña et al. 2018
    #     nc, ns, nt = data_hilbert.shape 
    #     ndat = np.divide(data_hilbert, abs(data_hilbert))
        
    #     plv = np.zeros([nc, nc, nt])
           
    #     for t in range(0, nt):   
    #         plv[:,:, t] = abs(ndat[:, :, t] @ ndat[:,:,t].conj().T) / ns
    
    #     # Computing RMS and store in PLV_all
    #     for t in range(0, plv.shape[-1]):        
    #         PLV_all_perm[t, idx_l, n_p] = np.sqrt(np.mean(np.square(plv[:, :, t])))    


    # Saving observed data for each participant
    saving_path = base_path + '/Sources/PLV_theta_parahipp_control/'
    os.chdir(saving_path)
        
    with open(('subj%s.pickle' % (val)), 'wb') as f:
        pickle.dump(PLV_all, f) 
    

    
    
####################################################################
# LOADING
####################################################################

subjlist = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

PLV_all = list()
META_all = list()

saving_path = base_path + '/Sources/PLV_theta_parahipp_control/'
data_path = base_path + '/ParticipantsData/'

for idx_s, val in enumerate(subjlist):

    os.chdir(saving_path)
    with open(('subj%s.pickle' % (val)), 'rb') as f:
      PLV_all.append(pickle.load(f))

    os.chdir(data_path + '/Subj' + str(val) + '/')
    meta = mne.read_epochs('Subj' + str(val) + '-epo.fif', 
                    verbose = False, preload = False).metadata
    META_all.append(meta.reset_index(drop = True)) 




####################################################################
# Preliminary unfair look to see if the are differences across tasks
####################################################################

# PLV_IMPL = np.zeros(shape = [len(subjlist), len(labels)])
# PLV_MEMO = np.zeros(shape = [len(subjlist), len(labels)])

# for idx_s, val in enumerate(subjlist):
#     PLV_IMPL[idx_s, :] = PLV_all[idx_s][META_all[idx_s].Task == 'Impl'].mean(axis = 0)
#     PLV_MEMO[idx_s, :] = PLV_all[idx_s][META_all[idx_s].Task == 'Memo'].mean(axis = 0)


# Diff = PLV_IMPL - PLV_MEMO

# import matplotlib.pyplot as plt

# labels_names = [i.name for i in labels]

# plt.boxplot(Diff, labels = labels_names)
# plt.xticks(rotation=90)
# plt.hlines(0, 0, 5, 'k', '--')

# stats_t, ps = stats.ttest_1samp(Diff, popmean = 0)  

# ## Correction for multiple comparisons

# from statsmodels.stats import multitest
# rej, ps_corr = multitest.fdrcorrection(ps, alpha=0.05, method='indep')

# sel_ps = [i for i, x in enumerate(ps_corr) if x < 0.05]
# sel_labels = [labels[i] for i in sel_ps]






####################################################################
# Preparing Data for Mixed effect models
####################################################################

# Creating gigantic dataframe with all data
for idx_s, val in enumerate(subjlist):
    for idx_l, label in enumerate(labels):
        META_all[idx_s][label.name] = PLV_all[idx_s][:,idx_l]


AllData = pd.concat(META_all)

# os.chdir(data_path)
# AllData.to_csv('AllData.csv')




SelData = AllData[['Subject', 'RT', 'Task','Cued_Side', 'Resp_Side', 'key', 'P_key', 'parahippocampal-lh', 
                     'parahippocampal-rh', 'Catch', 'acc' ]]


SelData.loc[SelData.Resp_Side == 'Left', 'contra-resp'] = SelData.loc[SelData.Resp_Side == 'Left', 'parahippocampal-rh']
SelData.loc[SelData.Resp_Side == 'Right', 'contra-resp'] = SelData.loc[SelData.Resp_Side == 'Right', 'parahippocampal-lh']

SelData.loc[SelData.Resp_Side == 'Left', 'ipsi-resp'] = SelData.loc[SelData.Resp_Side == 'Left', 'parahippocampal-lh']
SelData.loc[SelData.Resp_Side == 'Right', 'ipsi-resp'] = SelData.loc[SelData.Resp_Side == 'Right', 'parahippocampal-rh']



SelData.loc[SelData.Cued_Side == 'Left', 'contra-cued'] = SelData.loc[SelData.Cued_Side == 'Left', 'parahippocampal-rh']
SelData.loc[SelData.Cued_Side == 'Right', 'contra-cued'] = SelData.loc[SelData.Cued_Side == 'Right', 'parahippocampal-lh']

SelData.loc[SelData.Cued_Side == 'Left', 'ipsi-cued'] = SelData.loc[SelData.Cued_Side == 'Left', 'parahippocampal-lh']
SelData.loc[SelData.Cued_Side == 'Right', 'ipsi-cued'] = SelData.loc[SelData.Cued_Side == 'Right', 'parahippocampal-rh']


# os.chdir(data_path)
# SelData.to_csv('SelData.csv')


####################################################################
# PLOT
####################################################################


import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt

SelData = SelData[SelData.acc == 1]
SelData = SelData[SelData.Catch == 0]
SelData.RT = pd.to_numeric(SelData.RT) * 1000



NewData_a = SelData[['RT', 'Task', 'Subject', 'contra-resp', 'contra-cued']]
NewData_a['Laterality'] = 'Contra'
NewData_a = NewData_a.rename(columns=({'contra-resp': 'resp', 'contra-cued': 'cued'}))

NewData_b = SelData[['RT', 'Task', 'Subject', 'ipsi-resp', 'ipsi-cued']]
NewData_b['Laterality'] = 'Ipsi'
NewData_b = NewData_b.rename(columns=({'ipsi-resp': 'resp', 'ipsi-cued': 'cued'}))

NewData = pd.concat([NewData_a, NewData_b])


# os.chdir(data_path)
# NewData.to_csv('PLV_no_filtering.csv')



##################################################################
# Stats
##################################################################

import pingouin as pg

aov = pg.rm_anova(dv='resp', within=['Task', 'Laterality'],
                  subject='Subject', data=NewData,
                  effsize="ng2")


aov = pg.rm_anova(dv='cued', within=['Task', 'Laterality'],
                  subject='Subject', data=NewData,
                  effsize="ng2")






######################################################
# Plots
###################################################### 

a = NewData.groupby(['Task', 'Subject'], as_index = False).mean()

# set the font globally
plt.rcParams.update({'font.family':'Arial'})

fig, (ax1) = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 5))

## plotting hand

v1 = pt.half_violinplot( x ='Task' , y = 'resp', data = a, palette = ['blue', 'white'], bw = .3, cut = 5, scale = "area", width = .6, inner = None, orient = 'v', alpha = 0.8, ax= ax1, linewidth = 0)

ax1.set_xticklabels(['Implementation', 'Memorization'], fontsize = 16)
ax1.set_ylabel('PLV', fontsize = 16)
ax1.set_xlabel('Task', fontsize = 16)
ax1.set_title('mPFC - parahippocampal ROIs', fontsize = 20, fontweight='bold', y=1.05)
# ax1.set_ylim([0.46, 0.58])

ax2 = ax1.twiny()


v2 = pt.half_violinplot( x ='Task' , y = 'resp', data = a, order = ['Memo', 'Impl'], palette = ['red', 'white'], bw = .3, cut = 5.,scale = "area", width = .6, inner = None, orient = 'v', alpha = 0.8, ax= ax2, linewidth = 0)

ax2.invert_xaxis()
ax2.set_xlabel('')
ax2.set_xticklabels('')
ax2.set_xticks([])

ax3 = ax1.twiny()

for i, s in enumerate(subjlist):
    dat = a[a.Subject == s]
    sns.pointplot(data = dat, x = 'Task', y = 'resp',
                 color = 'lightgray', ax = ax3, dodge = True)


sns.pointplot(data = a, x = 'Task', y = 'resp', units = 'Subject', capsize = .1,
                 color = 'dimgray', ax = ax3, dodge = True)

# sns.boxplot(data = a, x = 'Task', y = 'hand', width = .1, saturation = .5,
#                  fliersize = 0, color = 'darkgray', ax = ax3, dodge = True)

ax3.set_xlabel('')
ax3.set_xticklabels('')
ax3.set_xticks([])





