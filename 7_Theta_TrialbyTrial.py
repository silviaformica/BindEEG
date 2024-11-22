# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:47:12 2021

@author: Silvia
"""


import mne
import numpy as np 
import pandas as pd
import os
import os.path as op
from scipy import signal, stats
import pickle
import matplotlib.pyplot as plt
import ptitprince as pt

base_path = 'F:/BindEEG/'
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


AllPower = list()


###############################################################################################################
## Extract theta power from mPFC - trial resolved
###############################################################################################################

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

#conditions = ['I_LL', 'I_LR', 'I_RL', 'I_RR', 'M_LL', 'M_LR', 'M_RL', 'M_RR']

subjlist = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

## START LOOPING ACROSS SUBJECTS

for idx_s, val in enumerate(subjlist):
    
    print('-----------------------------------\n-----------------------------------')
    print('Processing partitipant ', str(val))
    print('-----------------------------------\n-----------------------------------')

    data_path = base_path + 'ParticipantsData/Subj' + str(val) + '/'
    os.chdir(data_path)
    subj = mne.read_epochs('Subj' + str(val) + '-epo.fif', verbose = False)
    
    # keeping only correct trials
#    subj = subj[subj.metadata['acc'] == 1].pick_types(eeg = True)
    subj = subj.pick_types(eeg = True)    


    ####################################################################
    # Computing subject-specific preliminaries for source reconstruction
    ####################################################################
    
    # # Check that the locations of EEG electrodes is correct with respect to MRI
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

    # downsampling to 128 Hz
    subj = subj.decimate(4)

    

    ####################################################################
    # 
    ####################################################################

    # Projecting filtered signal on the sources
    stc_source = apply_inverse_epochs(subj, inv, lambda2,
    method, pick_ori="normal", verbose = False, return_generator = False)  
           
        
    # Looping across trials and extracting values for mPFC  
    # For tfr_array function it needs to be trials x vertices x time
    data_mPFC = np.zeros(shape = [ len(stc_source), stc_source[0].in_label(mPFC).data.shape[0], 
                                  len(stc_source[0].times)])      
    for t in range(len(stc_source)):
        data_mPFC[t, :, :] = stc_source[t].in_label(mPFC).data   
        
    
    # computing power and averaging across frequencies and vertices
    power = mne.time_frequency.tfr_array_morlet(data_mPFC, sfreq = subj.info['sfreq'],
                                               freqs = [3, 4, 5, 6, 7], n_cycles = 3,
                                               output = 'power', verbose = True)
    

    ## TODO
    ## A voler essere pignoli qui dovrei estrarre la first PCA
    ## invece che fare l'average across vertices
    
    
    AllPower.append(power.mean(axis = 2).mean(axis = 1))
    
        
os.chdir(base_path + '/ParticipantsData')    
with open("AllPower.txt", "wb") as fp:   #Pickling
    pickle.dump(AllPower, fp)    
    
        
####################################################################
# LOADING
####################################################################

subjlist = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

META_all = list()

data_path = base_path + '/ParticipantsData'
os.chdir(base_path + '/ParticipantsData')    

with open("AllPower.txt", "rb") as fp:   # Unpickling
    AllPower = pickle.load(fp)

# drop pp 3 from AllPower
del AllPower[2]


for idx_s, val in enumerate(subjlist):

    os.chdir(data_path + '/Subj' + str(val) + '/')
    meta = mne.read_epochs('Subj' + str(val) + '-epo.fif', 
                    verbose = False, preload = False).metadata
    META_all.append(meta) 


## TODO
## Decide on time window
## Decide on z-scoring

# Timepoints at which the signal will be cut

times = mne.read_epochs('Subj' + str(val) + '-epo.fif', 
                    verbose = False, preload = False).decimate(4).times

# These values are the boundaries of the significant cluster in theta
val0 = 0.54
val18 = 0.8
idx0, tp1 = find_nearest_tp(times, val0)
idx18, tp3 = find_nearest_tp(times, val18) 


for idx_s, val in enumerate(subjlist):

    META_all[idx_s]['mean_theta'] = AllPower[idx_s][:, idx0: idx18].mean(axis = 1)
    META_all[idx_s]['max_theta'] = AllPower[idx_s][:, idx0: idx18].max(axis = 1)
    META_all[idx_s]['median_theta'] = np.median(AllPower[idx_s][:, idx0: idx18], axis = 1)
   
    
for idx_s, val in enumerate(subjlist):

    META_all[idx_s]['mean_theta_zscore'] = stats.zscore(META_all[idx_s]['mean_theta'])
    META_all[idx_s]['max_theta_zscore'] = stats.zscore(META_all[idx_s]['max_theta'])
    META_all[idx_s]['median_theta_zscore'] =  stats.zscore(META_all[idx_s]['median_theta'])
#    META_all[idx_s]['RT_zscore'] =  stats.zscore(META_all[idx_s]['RT'].astype(float))
 
AllData = pd.concat(META_all)

AllData['mean_theta_zscore_all'] = stats.zscore(AllData['mean_theta'])
AllData['max_theta_zscore_all'] = stats.zscore(AllData['max_theta'])
AllData['median_theta_zscore_all'] = stats.zscore(AllData['median_theta'])
#AllData['RT_zscore_all'] = stats.zscore(AllData['RT'].astype(float))


os.chdir(data_path)
AllData.to_csv('ThetaData.csv')






## NO FILTERING
AllData = AllData[(AllData['acc'] == 1) & (AllData['Catch'] == 0)]
AllData.loc[:, 'RT'] = list(pd.to_numeric(AllData.RT))

dx="Task"; dy="mean_theta"; dhue = "Task"; ort="v"; pal = ['red', 'blue']; sigma = .2
df = AllData
#The Doge Flag
f, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(12, 5))
ax1=pt.RainCloud(x = 'Subject', y = dy, hue = dhue, data = df, palette = pal, bw = sigma,
                 width_viol = .7, ax = ax1, orient = ort , alpha = .65, dodge = True)
ax2=pt.RainCloud(x = 'Subject', y = 'RT', hue = dhue, data = df, palette = pal, bw = sigma,
                 width_viol = .7, ax = ax2, orient = ort , alpha = .65, dodge = True)

ax1.set_title('No filtering')
ax1.legend('')

os.chdir(data_path)
AllData.to_csv('ThetaData_nofilt.csv')




# ## FILTERING 1.5 std group level


# mean = AllData['mean_theta'].mean()
# std = (AllData['mean_theta']).std()   

# check = (AllData['mean_theta'] > mean - 1.5*std) & (AllData['mean_theta'] < mean + 1.5*std)

# # impl
# medianRTimpl = AllData[AllData['Task'] == 'Impl']['RT'].mean()
# MADRTimpl = (AllData[AllData['Task'] == 'Impl']['RT']).std()    

# # memo
# medianRTmemo = AllData[AllData['Task'] == 'Memo']['RT'].mean()
# MADRTmemo = (AllData[AllData['Task'] == 'Memo']['RT']).std()    


# checkrtimpl = (AllData['Task'] == 'Impl') & (AllData['RT'] > medianRTimpl - 1.5*MADRTimpl) & (AllData['RT'] < medianRTimpl + 1.5*MADRTimpl)

# checkrtmemo = (AllData['Task'] == 'Memo' ) & (AllData['RT'] > medianRTmemo - 1.5*MADRTmemo) & (AllData['RT'] < medianRTmemo + 1.5*MADRTmemo)
  


# AllData_filt = AllData.loc[(check & (checkrtimpl | checkrtmemo)), :]


# dx="Task"; dy="mean_theta"; dhue = "Task"; ort="v"; pal = ['red', 'blue']; sigma = .2
# df = AllData_filt


# f, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(12, 5))
# ax1=pt.RainCloud(x = 'Subject', y = dy, hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax1, orient = ort , alpha = .65, dodge = True)
# ax2=pt.RainCloud(x = 'Subject', y = 'RT', hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax2, orient = ort , alpha = .65, dodge = True)

# ax1.set_title('1.5 std for mean_theta and RTs')
# ax1.legend('')

# os.chdir(data_path)
# AllData_filt.to_csv('ThetaData_15group.csv')







# ######
# ## Trying filtering per pp

# Filt = list()

# for idx, subj in enumerate(META_all):
    
#     subj = subj[(subj['acc'] == 1) & (subj['Catch'] == 0)]
#     subj.loc[:, 'RT'] = list(pd.to_numeric(subj.RT))
    
#     mean = subj['mean_theta'].mean()
#     std = subj['mean_theta'].std()
    
#     check = (subj['mean_theta'] > mean - 3*std) & (subj['mean_theta'] < mean + 3*std)
    
#     subj_filt = subj.loc[check, :]

#     Filt.append(subj_filt)


# Filt = pd.concat(Filt)


# dx="Task"; dy="mean_theta"; dhue = "Task"; ort="v"; pal = ['red', 'blue']; sigma = .2
# df = Filt

# f, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(12, 5))
# ax1=pt.RainCloud(x = 'Subject', y = dy, hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax1, orient = ort , alpha = .65, dodge = True)
# ax2=pt.RainCloud(x = 'Subject', y = 'RT', hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax2, orient = ort , alpha = .65, dodge = True)

# ax1.set_title('3 std per participant for mean_theta')
# ax1.legend('')


# os.chdir(data_path)
# Filt.to_csv('ThetaData_thetafiltered3sd.csv')





## 3std mean_theta and Rts

Filt = list()
disc_theta = list()
disc_impl = list()
disc_memo = list()
len_subj = list()
len_impl = list()
len_memo = list()

for idx, subj in enumerate(META_all):
    
    subj = subj[(subj['acc'] == 1) & (subj['Catch'] == 0)]
    len_subj.append(len(subj))
    len_impl.append(len(subj[subj.Task == 'Impl']))
    len_memo.append(len(subj[subj.Task == 'Memo']))
   
    subj.loc[:, 'RT'] = list(pd.to_numeric(subj.RT))
    
    mean = subj['mean_theta'].mean()
    std = subj['mean_theta'].std()
    
    check = (subj['mean_theta'] > mean - 3*std) & (subj['mean_theta'] < mean + 3*std)
    
    disc_theta.append(len(check) - check.sum())
    
    # impl
    medianRTimpl = subj[subj['Task'] == 'Impl']['RT'].mean()
    MADRTimpl = (subj[subj['Task'] == 'Impl']['RT']).std()    
    
    # memo
    medianRTmemo = subj[subj['Task'] == 'Memo']['RT'].mean()
    MADRTmemo = (subj[subj['Task'] == 'Memo']['RT']).std()    
    
    
    checkrtimpl = (subj['Task'] == 'Impl') & (subj['RT'] > medianRTimpl - 3*MADRTimpl) & (subj['RT'] < medianRTimpl + 3*MADRTimpl)
    
    disc_impl.append((subj['Task'] == 'Impl').sum() - checkrtimpl.sum())
    
    checkrtmemo = (subj['Task'] == 'Memo' ) & (subj['RT'] > medianRTmemo - 3*MADRTmemo) & (subj['RT'] < medianRTmemo + 3*MADRTmemo)
      
    disc_memo.append((subj['Task'] == 'Memo').sum() - checkrtmemo.sum())

    
    subj_filt = subj.loc[(check & (checkrtimpl | checkrtmemo)), :]
    
    Filt.append(subj_filt)


Filt = pd.concat(Filt)


# dx="Task"; dy="mean_theta"; dhue = "Task"; ort="v"; pal = ['red', 'blue']; sigma = .2
# df = Filt

# f, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(12, 5))
# ax1=pt.RainCloud(x = 'Subject', y = dy, hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax1, orient = ort , alpha = .65, dodge = True, move = .25)
# ax2=pt.RainCloud(x = 'Subject', y = 'RT', hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax2, orient = ort , alpha = .65, dodge = True, move = .25)

# ax1.set_title('3 std per participant for mean_theta and RTs')
# ax1.legend('')
# f.tight_layout()


os.chdir(data_path)
Filt.to_csv('ThetaData_filtered3sd.csv')


print('Removed for theta power: %s  (±%s)' % (np.around(np.stack(disc_theta).mean(), decimals = 2), np.around(np.stack(disc_theta).std(), decimals = 2)))

print('Proportion: %s%%' % np.around(((np.stack(disc_theta)  * 100) / np.stack(len_subj)).mean(), decimals = 2))

print('Removed for impl rts: %s  (±%s)' % (np.around(np.stack(disc_impl).mean(), decimals = 2) ,    np.around(np.stack(disc_impl).std(), decimals = 2)))

print('Proportion: %s%%' % np.around(((np.stack(disc_impl)  * 100) / np.stack(len_impl)).mean(), decimals = 2))

print('Removed for memo rts: %s  (±%s)' % (np.around(np.stack(disc_memo).mean(), decimals = 2) ,    np.around(np.stack(disc_memo).std(), decimals = 2)))

print('Proportion: %s%%' % np.around(((np.stack(disc_memo)  * 100) / np.stack(len_memo)).mean(), decimals = 2))



# #### MAD


# Filt = list()

# for idx, subj in enumerate(META_all):
    
#     subj = subj[(subj['acc'] == 1) & (subj['Catch'] == 0)]
#     subj.loc[:, 'RT'] = pd.to_numeric(subj['RT'])
    
#     median = subj['mean_theta'].median()
#     MAD = stats.median_abs_deviation(subj['mean_theta'])   
    
#     check = (subj['mean_theta'] > median - 3*MAD) & (subj['mean_theta'] < median + 3*MAD)
    

#     # impl
#     medianRTimpl = subj[subj['Task'] == 'Impl']['RT'].median()
#     MADRTimpl = stats.median_abs_deviation(subj[subj['Task'] == 'Impl']['RT'])   

#     # memo
#     medianRTmemo = subj[subj['Task'] == 'Memo']['RT'].median()
#     MADRTmemo = stats.median_abs_deviation(subj[subj['Task'] == 'Memo']['RT'])   
    
    
#     checkrtimpl = (subj['Task'] == 'Impl') & (subj['RT'] > medianRTimpl - 3*MADRTimpl) & (subj['RT'] < medianRTimpl + 3*MADRTimpl)
    
#     checkrtmemo = (subj['Task'] == 'Memo' ) & (subj['RT'] > medianRTmemo - 3*MADRTmemo) & (subj['RT'] < medianRTmemo + 3*MADRTmemo)
   

    
#     subj_filt = subj.loc[(check & (checkrtimpl | checkrtmemo)), :]

#     # subj_filt = subj.loc[check, :]


#     Filt.append(subj_filt)


# Filt = pd.concat(Filt)

# dx="Task"; dy="mean_theta"; dhue = "Task"; ort="v"; pal = ['red', 'blue']; sigma = .2
# df = Filt

# f, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(12, 5))
# ax1=pt.RainCloud(x = 'Subject', y = dy, hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax1, orient = ort , alpha = .65, dodge = True)
# ax2=pt.RainCloud(x = 'Subject', y = 'RT', hue = dhue, data = df, palette = pal, bw = sigma,
#                  width_viol = .7, ax = ax2, orient = ort , alpha = .65, dodge = True)

# ax1.set_title('3 MAD per participant for mean_theta and RTs')
# ax1.legend('')






# os.chdir(data_path)
# Filt.to_csv('ThetaData_filtered3MAD.csv')
