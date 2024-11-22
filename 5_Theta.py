# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:20:48 2021

@author: Silvia Formica
"""


import mne
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as op
from mne.datasets import fetch_fsaverage
import seaborn as sns
from mne import stats
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test, summarize_clusters_stc
from mne.stats import f_mway_rm, f_threshold_mway_rm, ttest_1samp_no_p, permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test
import pickle 
from scipy import stats as stats

base_path = 'F:/BindEEG/'
os.chdir(base_path)

###############################################################################################################
## Importing anatomy template and specific functions
###############################################################################################################

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation

src_name = op.join(fs_dir, 'bem', 'fsaverage-oct-6-src.fif')
src = mne.read_source_spaces(src_name, verbose = True)

###############################################################################################################
# Loading and plotting results
###############################################################################################################

os.chdir(base_path + '/Sources/theta_whole_brain')

subjlist = [1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

Impl_LL = list()
Impl_LR = list()
Impl_RL = list()
Impl_RR = list()
Memo_LL = list()
Memo_LR = list()
Memo_RL = list()
Memo_RR = list()


for idx, val in enumerate(subjlist):    
    Impl_LL.append(mne.read_source_estimate('Impl_LL_%s' % val))
    Impl_LR.append(mne.read_source_estimate('Impl_LR_%s' % val))
    Impl_RL.append(mne.read_source_estimate('Impl_RL_%s' % val))
    Impl_RR.append(mne.read_source_estimate('Impl_RR_%s' % val))
    Memo_LL.append(mne.read_source_estimate('Memo_LL_%s' % val))
    Memo_LR.append(mne.read_source_estimate('Memo_LR_%s' % val))
    Memo_RL.append(mne.read_source_estimate('Memo_RL_%s' % val))
    Memo_RR.append(mne.read_source_estimate('Memo_RR_%s' % val))

    
###############################################################################################################
# Extracting time courses from label of choice
###############################################################################################################
   
# Following line is to download the dataset if you don't have it already
# mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
#                                         verbose=True)

# Choose parcellation atlas:
# aparc = Desikan-Killiany - 34 ROIs/hemisphere
# HCPMMP1 = multimodal parcellation - 181 ROIs/hemisphere
# HCPMMP1_combined - 23 ROIs/hemisphere

labels = mne.read_labels_from_annot(
'fsaverage', 'aparc', 'both', subjects_dir=subjects_dir)    

# # visualize parcellation and ROI of interest. To see all labels names check the variable "label"
# Brain = mne.viz.get_brain_class()
# brain = Brain('fsaverage', 'split', 'inflated', subjects_dir=subjects_dir,
#           cortex='low_contrast', background='white', size=(800, 600))
# brain.add_annotation('aparc')


label_to_plot = [label for label in labels if label.name == 'caudalanteriorcingulate-lh'][0]
# brain.add_label(label_to_plot, borders=False)  

label_to_plot = [label for label in labels if label.name == 'caudalanteriorcingulate-rh'][0]
# brain.add_label(label_to_plot, borders=False)  

# Choose ROI - bilaterally
label_L = [label for label in labels if label.name == 'caudalanteriorcingulate-lh'][0]
label_R = [label for label in labels if label.name == 'caudalanteriorcingulate-rh'][0]

label = mne.BiHemiLabel(label_L, label_R)

# Extracting time course

Impl = list()
Memo = list()

for idx, val in enumerate(subjlist):

    this_Impl = list()
    this_Memo = list()    


    this_Impl.append(Impl_LL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Impl.append(Impl_LR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Impl.append(Impl_RL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Impl.append(Impl_RR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Impl = np.mean(np.stack(this_Impl), axis = 0)
    Impl.append(this_Impl)
    
    this_Memo.append(Memo_LL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Memo.append(Memo_LR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Memo.append(Memo_RL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Memo.append(Memo_RR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    this_Memo = np.mean(np.stack(this_Memo), axis = 0)
    Memo.append(this_Memo)  



###############################################################################################################
# STATS
###############################################################################################################

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations

X = np.stack(Impl) - np.stack(Memo)
X = X[:, 0, -231:]

pthresh = 0.05  # set threshold rather high to save some time
t_threshold = -stats.distributions.t.ppf(pthresh / 2., len(subjlist) - 1)

tail = 1 # 
n_permutations = 100000

T_obs_A, clusters_A, cluster_p_values_A, h0 = permutation_cluster_1samp_test(
    X, stat_fun = mne.stats.ttest_1samp_no_p, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')



# P = 0.033

# Computing effect size

for i_c, c in enumerate(clusters_A):
    c = c[0]
    if cluster_p_values_A[i_c] <= 0.05:
        idx = [c.start, c.stop]
        
X1 = np.stack(Impl).squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)
X2 = np.stack(Memo).squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)

CohenD = np.mean(X1 - X2) / np.std(X1 - X2)
# d = 0.57

# plt.figure()
# plt.plot([X1, X2], 'o-')






###############################################################################################################
# PLOTS
###############################################################################################################


data = pd.DataFrame()
data['value'] = (np.squeeze(Impl).flatten() - np.squeeze(Memo).flatten())
data['timepoint'] = np.tile(Impl_LL[0].copy().crop(-0.1, 1.8).times, len(subjlist))*1000
#data['Task'] = ['Implementation'] * Impl_CvsI.shape[1] * Impl_CvsI.shape[0] + ['Memorization'] * Memo_CvsI.shape[1] * Memo_CvsI.shape[0]


times_plot = (Impl_LL[0].copy().crop(-0.1, 1.8).times)*1000
neg_timep = len(Impl_LL[0].copy().crop(-0.1, 0).times)

## finding exact timepoints of cluster
b = times_plot[neg_timep-1:]
b[clusters_A[0][0].start]
b[clusters_A[0][0].stop]
# Plot with significance for main effect

fig, ax = plt.subplots()
sns.lineplot(x="timepoint", y="value",
             color = 'black', ci = 68, 
             data=data)
plt.axhline(0, color = 'k', ls='--')
plt.axvline(0, color = 'k', ls='--')

for i_c, c in enumerate(clusters_A):
    if cluster_p_values_A[i_c] <= 0.05:
        
        h = plt.axvspan(times_plot[clusters_A[i_c][0].start + neg_timep] , times_plot[clusters_A[i_c][0].stop + neg_timep],
                        color='grey', alpha = 0.3)
                        
plt.xlabel('Time (ms)')
plt.ylabel('Power (au)')
plt.title('Implementation vs Memorization - Caudalanteriorcingulate ROI')



###############################################################################################################
# Extracting at pp level
###############################################################################################################

times_plot = (Impl_LL[0].copy().crop(-0.1, 1.8).times)*1000
neg_timep = len(Impl_LL[0].copy().crop(-0.1, 0).times)

data = pd.DataFrame()
data['value'] = np.hstack((np.squeeze(Impl).flatten(),  np.squeeze(Memo).flatten()))
data['timepoint'] = np.tile(Impl_LL[0].copy().crop(-0.1, 1.8).times, len(subjlist)*2)*1000
data['Task'] = ['Implementation'] * int(len(data)/2) + ['Memorization'] * int(len(data)/2) 

fig, ax = plt.subplots()
sns.lineplot(x="timepoint", y="value", hue = 'Task',
             palette = ['blue', 'red'], ci = 68, 
             data=data)

plt.axvline(0, color = 'k', ls='--')

for i_c, c in enumerate(clusters_A):
    if cluster_p_values_A[i_c] <= 0.05:
        
        h = plt.axvspan(times_plot[clusters_A[0][0].start + neg_timep] , times_plot[clusters_A[0][0].stop + neg_timep],
                        color='grey', alpha = 0.3)

plt.xlabel('Time (ms)')
plt.ylabel('Power (au)')
plt.title('Implementation vs Memorization - mPFC ROI')




# cluster extent
tps = np.arange(clusters_A[0][0].start, clusters_A[0][0].stop + 1)

# taking mean power in the cluster
Impl_cropped = np.stack(Impl).squeeze()[:, -231:]
Memo_cropped = np.stack(Memo).squeeze()[:, -231:]

impl_avg = Impl_cropped[:, tps].mean(axis = 1)
memo_avg = Memo_cropped[:, tps].mean(axis = 1)

import pickle
os.chdir('D:/BindEEG/ParticipantsData/')

with open('BrainBehavior', 'rb') as f:
     Behaviorals = pickle.load(f)
 
Behaviorals['theta_impl'] = impl_avg
Behaviorals['theta_memo'] = memo_avg

Behaviorals.to_csv('BrainBehavior_theta.csv')


        
###############################################################################################################
# ANOVA
###############################################################################################################


# Extracting time course

Impl_CuedLeft_RespLeft = list()
Impl_CuedLeft_RespRight = list()
Impl_CuedRight_RespLeft = list()
Impl_CuedRight_RespRight = list()
Memo_CuedLeft_RespLeft = list()
Memo_CuedLeft_RespRight = list()
Memo_CuedRight_RespLeft = list()
Memo_CuedRight_RespRight = list()


for idx, val in enumerate(subjlist):
    
    Impl_CuedLeft_RespLeft.append(Impl_LL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Impl_CuedLeft_RespRight.append(Impl_LR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Impl_CuedRight_RespLeft.append(Impl_RL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Impl_CuedRight_RespRight.append(Impl_RR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedLeft_RespLeft.append(Memo_LL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedLeft_RespRight.append(Memo_LR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedRight_RespLeft.append(Memo_RL[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedRight_RespRight.append(Memo_RR[idx].crop(-0.1, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))


# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations
X = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft, Impl_CuedRight_RespRight,
              Memo_CuedLeft_RespLeft, Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft, Memo_CuedRight_RespRight ])
X = np.transpose(X, [0, 1, 3, 2]) # conditions x subj x time
X = np.squeeze(X)
X = X[:, :, -231:]
X_list = list(X)


# rmANOVA settings
# effect A is the one repeating slowest (here: Task)
# effect B is the one repeating faster (here: Cued Side)
# effect C is the fastest (here: Resp side)
# interactions A:B, A:C, B:C, A:B:C

factor_levels = [2, 2, 2]
effects_labels = ['Impl_LL', 'Impl_LR', 'Impl_RL', 'Impl_RR', 'Memo_LL', 'Memo_LR', 'Memo_RL', 'Memo_RR']
n_conditions = len(effects_labels)
n_subj = len(subjlist)
times = Impl_LL[0].copy().crop(0, 1.8).times*1000
n_times = len(Impl_LL[0].copy().crop(0, 1.8).times)


# Main effect of TASK

effects = 'A'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here


# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_A = \
    permutation_cluster_test(X_list, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)


## Main effect of CUED SIDE

effects = 'B'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                      effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_B = \
    permutation_cluster_test(X_list, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)



## Main effect of RESP SIDE

effects = 'C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                      effects=effects, return_pvals=False)[0]      #don't need p_values here
# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_C = \
    permutation_cluster_test(X_list, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)



## Interaction TASK * CUED SIDE

effects = 'A:B'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_AB = \
    permutation_cluster_test(X_list, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)



## Interaction TASK * RESP SIDE

effects = 'A:C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)


print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_AC = \
    permutation_cluster_test(X_list, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations) 

## Interaction CUED SIDE * CUED SIDE

effects = 'B:C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)


print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_BC = \
    permutation_cluster_test(X_list, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)


## Interaction TASK * CUED SIDE * CUED SIDE

effects = 'A:B:C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)


print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_ABC = \
    permutation_cluster_test(X_list, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)













###############################################################################################################
# plots on sources
###############################################################################################################

Impl = list()
Memo = list()

for idx, val in enumerate(subjlist):

    this_Impl = list()
    this_Memo = list()    


    this_Impl.append(Impl_LL[idx].copy().crop(-0.1, 1.8).data)
    this_Impl.append(Impl_LR[idx].copy().crop(-0.1, 1.8).data)
    this_Impl.append(Impl_RL[idx].copy().crop(-0.1, 1.8).data)
    this_Impl.append(Impl_RR[idx].copy().crop(-0.1, 1.8).data)
    this_Impl = np.mean(np.stack(this_Impl), axis = 0)
    Impl.append(this_Impl)
    
    this_Memo.append(Memo_LL[idx].copy().crop(-0.1, 1.8).data)
    this_Memo.append(Memo_LR[idx].copy().crop(-0.1, 1.8).data)
    this_Memo.append(Memo_RL[idx].copy().crop(-0.1, 1.8).data)
    this_Memo.append(Memo_RR[idx].copy().crop(-0.1, 1.8).data)
    this_Memo = np.mean(np.stack(this_Memo), axis = 0)
    Memo.append(this_Memo)  


Impl_stc = Impl_LL[0].copy()
Impl_stc.data = np.mean(np.stack(Impl), axis = 0)

Memo_stc = Memo_LL[0].copy()
Memo_stc.data = np.mean(np.stack(Memo), axis = 0)

Diff = Impl_stc - Memo_stc

# Diff.plot(subject = 'fsaverage', hemi='split', subjects_dir=subjects_dir,
#                background = 'white', 
#           size=(800, 600))



mne.viz.set_3d_backend("mayavi")

#from mayavi import mlab

#fig = mlab.figure(size=(300, 300))
brain = Diff.plot(
    subject='fsaverage',
    hemi='split', 
    background='white',
    foreground='black',
    time_label='',
    initial_time=0.7,
    smoothing_steps=10,
    alpha = 1,
    clim=dict(kind='value', pos_lims=[1.1, 1.2, 2]),
    views = ['lat', 'med'])








###############################################################################################################
# CONGRUENCY CHECKS ON TIME COURSES
###############################################################################################################




# Extracting time course

Impl_CuedLeft_RespLeft = list()
Impl_CuedLeft_RespRight = list()
Impl_CuedRight_RespLeft = list()
Impl_CuedRight_RespRight = list()
Memo_CuedLeft_RespLeft = list()
Memo_CuedLeft_RespRight = list()
Memo_CuedRight_RespLeft = list()
Memo_CuedRight_RespRight = list()


for idx, val in enumerate(subjlist):
    
    Impl_CuedLeft_RespLeft.append(Impl_LL[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Impl_CuedLeft_RespRight.append(Impl_LR[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Impl_CuedRight_RespLeft.append(Impl_RL[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Impl_CuedRight_RespRight.append(Impl_RR[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedLeft_RespLeft.append(Memo_LL[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedLeft_RespRight.append(Memo_LR[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedRight_RespLeft.append(Memo_RL[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))
    Memo_CuedRight_RespRight.append(Memo_RR[idx].crop(0, 1.8).extract_label_time_course(label, src, mode = 'pca_flip'))




###############################################################################################################
# Congruent vs incongruent Impl
###############################################################################################################

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations
I_congr = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedRight_RespRight])
I_congr = np.mean(I_congr, axis = 0)
                
I_incongr = np.stack([Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft])
I_incongr = np.mean(I_incongr, axis = 0)

I_contr = I_incongr - I_congr

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
I_contr = np.transpose(I_contr, [0, 2, 1]).squeeze()

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_I = \
    permutation_cluster_1samp_test(I_contr, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)



###############################################################################################################
# Congruent vs incongruent Memo
###############################################################################################################


M_congr = np.stack([Memo_CuedLeft_RespLeft, Memo_CuedRight_RespRight])
M_congr = np.mean(M_congr, axis = 0)
                
M_incongr = np.stack([Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft])
M_incongr = np.mean(M_incongr, axis = 0)

M_contr = M_incongr - M_congr

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
M_contr = np.transpose(M_contr, [0, 2, 1]).squeeze()

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_M = \
    permutation_cluster_1samp_test(
    M_contr, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


# P = 0.168

# Computing effect size

for i_c, c in enumerate(clusters):
    c = c[0]
    # if cluster_p_values[i_c] <= 0.05:
    idx = [c.start, c.stop]
        
X1 = M_incongr.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)
X2 = M_congr.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)

CohenD = np.mean(X1 - X2) / np.std(X1 - X2)
# d = 0.46

# plt.figure()
# plt.plot([X1, X2], 'o-')




###############################################################################################################
# Between Tasks
###############################################################################################################

Diff = I_contr - M_contr
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_IM = \
    permutation_cluster_1samp_test(Diff, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)



###############################################################################################################
# Congruent vs incongruent across tasks
###############################################################################################################

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations
congr = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedRight_RespRight, Memo_CuedLeft_RespLeft, Memo_CuedRight_RespRight])
congr = np.mean(congr, axis = 0)
                
incongr = np.stack([Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft, Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft])
incongr = np.mean(incongr, axis = 0)

contr = incongr - congr

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
contr = np.transpose(contr, [0, 2, 1]).squeeze()

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_congruency = \
    permutation_cluster_1samp_test(I_contr, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)




###############################################################################################################
# Impl vs Memo only in congruent trials
###############################################################################################################

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations
I_congr = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedRight_RespRight])
I_congr = np.mean(I_congr, axis = 0)

M_congr = np.stack([Memo_CuedLeft_RespLeft, Memo_CuedRight_RespRight])
M_congr = np.mean(M_congr, axis = 0)

Diff = I_congr - M_congr
Diff = np.transpose(Diff, [0, 2, 1]).squeeze()

p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_IvsM_congr = \
    permutation_cluster_1samp_test(Diff, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True, out_type='mask')



# P = 0.033

# Computing effect size

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        idx = [c.start, c.stop]
        
X1 = I_congr.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)
X2 = M_congr.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)

CohenD = np.mean(X1 - X2) / np.std(X1 - X2)
# d = 0.54

# plt.figure()
# plt.plot([X1, X2], 'o-')






###############################################################################################################
# Impl vs Memo only in incongruent trials
###############################################################################################################


I_incongr = np.stack([Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft])
I_incongr = np.mean(I_incongr, axis = 0)
           
M_incongr = np.stack([Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft])
M_incongr = np.mean(M_incongr, axis = 0)

Diff = I_incongr - M_incongr
Diff = np.transpose(Diff, [0, 2, 1]).squeeze()

p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 10000 # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_IvsM_incongr = \
    permutation_cluster_1samp_test(Diff, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True, out_type='mask')


# P = 0.125

# Computing effect size

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.13:
        idx = [c.start, c.stop]
        
X1 = I_incongr.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)
X2 = M_incongr.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)

CohenD = np.mean(X1 - X2) / np.std(X1 - X2)
# d = 0.43

# plt.figure()
# plt.plot([X1, X2], 'o-')




###############################################################################################################
# Congruent vs INcongruent across tasks
###############################################################################################################


congr = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedRight_RespRight, Memo_CuedLeft_RespLeft, Memo_CuedRight_RespRight])
congr = np.mean(congr, axis = 0)
           
incongr = np.stack([Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft, Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft])
incongr = np.mean(incongr, axis = 0)

Diff = incongr - congr
Diff = np.transpose(Diff, [0, 2, 1]).squeeze()

p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_congr_vs_incongr = \
    permutation_cluster_1samp_test(Diff, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold= t_threshold, buffer_size=None,
                                       verbose=True)














###############################################################################################################
# STATS on sources
###############################################################################################################

Impl_CuedLeft_RespLeft = list()
Impl_CuedLeft_RespRight = list()
Impl_CuedRight_RespLeft = list()
Impl_CuedRight_RespRight = list()
Memo_CuedLeft_RespLeft = list()
Memo_CuedLeft_RespRight = list()
Memo_CuedRight_RespLeft = list()
Memo_CuedRight_RespRight = list()


for idx, val in enumerate(subjlist): 


    Impl_CuedLeft_RespLeft.append(Impl_LL[idx].copy().crop(0, 1.8).data)
    Impl_CuedLeft_RespRight.append(Impl_LR[idx].copy().crop(0, 1.8).data)
    Impl_CuedRight_RespLeft.append(Impl_RL[idx].copy().crop(0, 1.8).data)
    Impl_CuedRight_RespRight.append(Impl_RR[idx].copy().crop(0, 1.8).data)
    Memo_CuedLeft_RespLeft.append(Memo_LL[idx].copy().crop(0, 1.8).data)
    Memo_CuedLeft_RespRight.append(Memo_LR[idx].copy().crop(0, 1.8).data)
    Memo_CuedRight_RespLeft.append(Memo_RL[idx].copy().crop(0, 1.8).data)
    Memo_CuedRight_RespRight.append(Memo_RR[idx].copy().crop(0, 1.8).data)






# as we only have one hemisphere we need only need half the adjacency
print('Computing adjacency.')
adjacency = mne.spatial_src_adjacency(src)


#######
## Adjust adjacency matrix to interhemispheric neighbours



dist = 0.02    # need to find the right threshold!

from scipy import sparse
from scipy.spatial.distance import cdist
adj = cdist(src[0]['rr'][src[0]['vertno']],
            src[1]['rr'][src[1]['vertno']])
adj = sparse.csr_matrix(adj <= dist, dtype=int)
empties = [sparse.csr_matrix((nv, nv), dtype=int) for nv in adj.shape]
adj = sparse.vstack([sparse.hstack([empties[0], adj]),
                     sparse.hstack([adj.T, empties[1]])])


# fig = plt.figure()
# ax = fig.add_subplot(111, facecolor='black')
# ax.plot(adj.col, adj.row, 's', color='white', ms=1)
# ax.set_xlim(0, adjacency.shape[1])
# ax.set_ylim(0, adjacency.shape[0])
# ax.set_aspect('equal')
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.invert_yaxis()
# ax.set_aspect('equal')
# ax.set_xticks([])
# ax.set_yticks([])

adj_tot = sparse.coo_matrix(adjacency + adj)

# fig = plt.figure()
# ax = fig.add_subplot(111, facecolor='black')
# ax.plot(adj_tot.col, adj_tot.row, 's', color='white', ms=1)
# ax.set_xlim(0, adjacency.shape[1])
# ax.set_ylim(0, adjacency.shape[0])
# ax.set_aspect('equal')
# for spine in ax.spines.values():
#     spine.set_visible(False)
# ax.invert_yaxis()
# ax.set_aspect('equal')
# ax.set_xticks([])
# ax.set_yticks([])


# mne.viz.set_3d_backend("pyvista")

# brain = mne.viz.Brain(subject_id='fsaverage', subjects_dir=subjects_dir, surf = 'inflated', hemi = 'both', units = 'm', alpha = 0, background = 'white')

# brain.add_foci(src[0]['rr'][src[0]['vertno']], hemi = 'lh', scale_factor = 0.1, color = 'red', alpha = 0.5)
# brain.add_foci(src[1]['rr'][src[1]['vertno']], hemi = 'rh', scale_factor = 0.1, color = 'green', alpha = 0.5)


# brain.add_foci(src[0]['rr'][src[0]['vertno'][np.unique(adj.row)[np.unique(adj.row)<4098]]], hemi = 'lh', scale_factor = 0.2, color = 'red')

# brain.add_foci(src[1]['rr'][src[1]['vertno'][np.unique(adj.row)[np.unique(adj.row)>4098]-4098]], hemi = 'rh', scale_factor = 0.2, color = 'green')








# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations
X = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft, Impl_CuedRight_RespRight,
              Memo_CuedLeft_RespLeft, Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft, Memo_CuedRight_RespRight ])
X = np.transpose(X, [0, 1, 3, 2]) # conditions x subj x time x space
X_list = list(X)


# rmANOVA settings
# effect A is the one repeating slowest (here: Task)
# effect B is the one repeating faster (here: Cued Side)
# effect C is the fastest (here: Resp side)
# interactions A:B, A:C, B:C, A:B:C

factor_levels = [2, 2, 2]
effects_labels = ['Impl_LL', 'Impl_LR', 'Impl_RL', 'Impl_RR', 'Memo_LL', 'Memo_LR', 'Memo_RL', 'Memo_RR']
n_conditions = len(effects_labels)
n_subj = len(subjlist)
times = Impl_LL[0].copy().crop(0, 1.8).times*1000
n_times = len(Impl_LL[0].copy().crop(0, 1.8).times)


# Main effect of TASK

effects = 'A'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 100  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_A = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)

with open('clu_A.pickle', 'wb') as f:
    pickle.dump(clu_A, f)

del(clu_A)


## Main effect of CUED SIDE

effects = 'B'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                      effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_B = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                  threshold=f_thresh, stat_fun=stat_fun,
                                  n_permutations=n_permutations)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

with open('clu_B.pickle', 'wb') as f:
    pickle.dump(clu_B, f)

del(clu_B)



## Main effect of RESP SIDE

effects = 'C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                      effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_C = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                  threshold=f_thresh, stat_fun=stat_fun,
                                  n_permutations=n_permutations)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

with open('clu_C.pickle', 'wb') as f:
    pickle.dump(clu_C, f)

del(clu_C)




## Interaction TASK * CUED SIDE

effects = 'A:B'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_AB = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


with open('clu_AB.pickle', 'wb') as f:
    pickle.dump(clu_AB, f)

del(clu_AB)


## Interaction TASK * RESP SIDE

effects = 'A:C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_AC = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


with open('clu_AC.pickle', 'wb') as f:
    pickle.dump(clu_AC, f)
    
del(clu_AC)    

## Interaction CUED SIDE * CUED SIDE

effects = 'B:C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_BC = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]


with open('clu_BC.pickle', 'wb') as f:
    pickle.dump(clu_BC, f)

del(clu_BC)


## Interaction TASK * CUED SIDE * CUED SIDE

effects = 'A:B:C'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_ABC = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)


with open('clu_ABC.pickle', 'wb') as f:
    pickle.dump(clu_ABC, f)

del(clu_ABC)



###############################################################################################################
# PLOTTING CLUSTERS
###############################################################################################################

# Loading Clusters

with open('clu_A.pickle', 'rb') as f:
      clu_A = pickle.load(f)

with open('clu_B.pickle', 'rb') as f:
      clu_B = pickle.load(f)
      
with open('clu_C.pickle', 'rb') as f:
      clu_C = pickle.load(f)

with open('clu_AB.pickle', 'rb') as f:
      clu_AB = pickle.load(f)

with open('clu_AC.pickle', 'rb') as f:
      clu_AC = pickle.load(f)

with open('clu_BC.pickle', 'rb') as f:
      clu_BC = pickle.load(f)

with open('clu_ABC.pickle', 'rb') as f:
      clu_ABC = pickle.load(f)


      
np.sort(clu_A[2])
np.sort(clu_B[2])
np.sort(clu_C[2])
np.sort(clu_AB[2])
np.sort(clu_AC[2])
np.sort(clu_BC[2])
np.sort(clu_ABC[2])



print('Visualizing clusters.')
fsave_vertices = src
# tstep = Impl_LL[0].copy().tstep * 1000

tstep = 7.8125

#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu_A, tstep=tstep, p_thresh=0.2,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration


brain = stc_all_cluster_vis.plot(subject='fsaverage', hemi = 'split',
                                 time_label='temporal extent (ms)',
                                 cortex='low_contrast', transparent = True)



mne.viz.set_3d_backend("mayavi")

brain = stc_all_cluster_vis.plot(
    subject='fsaverage',
    hemi='split', 
    background='white',
    foreground='black',
    time_label='',
    initial_time=1,
    smoothing_steps=10,
    alpha = 1,
    transparent = True,
    clim=dict(kind='value', lims=[0, 1, np.max(stc_all_cluster_vis.data)]))



# time points of the almost significant cluster
time_points_clu = np.unique(clu_BC[1][0][0])

times = Impl_LL[0].copy().crop(0, 1.8).times

times_clu = times[time_points_clu]

times_bi = np.isin(times, times_clu, assume_unique = True)
series = pd.Series(times_bi)  

data = pd.DataFrame()
data['value'] = series
data['timepoint'] = times*1000

fig, ax = plt.subplots()
sns.lineplot(x="timepoint", y="value",
             data=data)
plt.xlabel('Time (ms)')








###############################################################################################################
# excluding deep sources
###############################################################################################################



labels_lh = mne.read_labels_from_annot(
'fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)    

labels_rh = mne.read_labels_from_annot(
'fsaverage', 'HCPMMP1_combined', 'rh', subjects_dir=subjects_dir)  


stc = Impl_LL[0]                 
drop_stc = stc.in_label(labels_lh[0] + labels_rh[0])
                               
All_vertices_lh = stc.vertices[0]                        
All_vertices_rh = stc.vertices[1]

exclude_lh = np.in1d(All_vertices_lh, drop_stc.vertices[0])
exclude_rh = np.in1d(All_vertices_rh, drop_stc.vertices[1])

spatial_exclude = np.concatenate([np.where(exclude_lh)[0], np.where(exclude_rh)[0] + len(stc.vertices[0])])


# Main effect of TASK


effects = 'A'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.01  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(len(subjlist), factor_levels, effects, pthresh)

threshold_tfce = dict(start=0, step=0.2)

tail = 1  # f-test, so tail > 0
n_permutations = 100  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_A = \
    spatio_temporal_cluster_test(X, adjacency=adj_tot, n_jobs=4,
                                 threshold=f_thresh, stat_fun=stat_fun, spatial_exclude = spatial_exclude,
                                 n_permutations=n_permutations)

print('Visualizing clusters.')
fsave_vertices = src
tstep = Impl_LL[0].copy().tstep * 1000
#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu_A, tstep=tstep, p_thresh=0.2,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration


brain = stc_all_cluster_vis.plot(subject='fsaverage', hemi = 'split',
                                 time_label='temporal extent (ms)',
                                 cortex='low_contrast', transparent = True,
                                 clim=dict(kind='value', lims=[0, 1, 1000]))


mne.viz.set_3d_backend("pyvista")














# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations
X_I = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft, Impl_CuedRight_RespRight])
X_I = np.mean(X_I, axis = 0)
                
X_M = np.stack([Memo_CuedLeft_RespLeft, Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft, Memo_CuedRight_RespRight])
X_M = np.mean(X_M, axis = 0)

X = X_I - X_M


#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X = np.transpose(X, [0, 2, 1])

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, adjacency=adj_tot, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)




print('Visualizing clusters.')
fsave_vertices = src

tstep = 7.8125


stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, p_thresh=0.05,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration


mne.viz.set_3d_backend("mayavi")

brain = stc_all_cluster_vis.plot(
    subject='fsaverage',
    hemi='split', 
    background='white',
    foreground='black',
    time_label='',
    initial_time=1,
    smoothing_steps=10,
    alpha = 1,
    transparent = True,
    clim=dict(kind='value', lims=[0, 1, np.max(stc_all_cluster_vis.data)]))



# time points of the almost significant cluster
time_points_clu = np.unique(clu[1][0][0])

times = Impl_LL[0].copy().crop(0, 1.8).times

times_clu = times[time_points_clu]

times_bi = np.isin(times, times_clu, assume_unique = True)
series = pd.Series(times_bi)  

data = pd.DataFrame()
data['value'] = series
data['timepoint'] = times*1000

fig, ax = plt.subplots()
sns.lineplot(x="timepoint", y="value",
             data=data)
plt.xlabel('Time (ms)')





