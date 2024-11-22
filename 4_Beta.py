# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:37:24 2021

@author: Carlos
"""

# %matplotlib qt

import mne
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as op
from mne.datasets import fetch_fsaverage
import seaborn as sns
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test, summarize_clusters_stc, permutation_cluster_1samp_test
from mne.stats import f_mway_rm, f_threshold_mway_rm
import pickle 
from scipy import spatial

base_path = 'D:/BindEEG/'
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
# Loading results
###############################################################################################################

os.chdir(base_path + '/Sources/beta_whole_brain')

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
    Impl_LL.append({'beta' : mne.read_source_estimate('Impl_LL_%s' % val)})
    Impl_LR.append({'beta' : mne.read_source_estimate('Impl_LR_%s' % val)})
    Impl_RL.append({'beta' : mne.read_source_estimate('Impl_RL_%s' % val)})
    Impl_RR.append({'beta' : mne.read_source_estimate('Impl_RR_%s' % val)})
    Memo_LL.append({'beta' : mne.read_source_estimate('Memo_LL_%s' % val)})
    Memo_LR.append({'beta' : mne.read_source_estimate('Memo_LR_%s' % val)})
    Memo_RL.append({'beta' : mne.read_source_estimate('Memo_RL_%s' % val)})
    Memo_RR.append({'beta' : mne.read_source_estimate('Memo_RR_%s' % val)})
    
    
    
###############################################################################################################
# Extracting time courses from label of choice
###############################################################################################################

MNI_hand_left = [(-44., -17., 49.)]
MNI_hand_right = [(44., -17., 49.)]


lefties = np.squeeze(mne.vertex_to_mni([src[0]['vertno']], hemis = 0, subject = 'fsaverage'))
righties = np.squeeze(mne.vertex_to_mni([src[1]['vertno']], hemis = 1, subject = 'fsaverage'))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (spatial.distance.cdist(array , value)).argmin()
    return idx, array[idx]

idx_left, vv_left = find_nearest(lefties, MNI_hand_left)
idx_right, vv_right = find_nearest(righties, MNI_hand_right)



## Growing labels for left and right hand

label_L = mne.grow_labels('fsaverage', seeds = src[0]['vertno'][idx_left], extents = 30, hemis = 0, names = 'HandLeft', surface = 'inflated')[0]
label_R = mne.grow_labels('fsaverage', seeds = src[1]['vertno'][idx_right], extents = 30, hemis = 1, names = 'HandRight', surface = 'inflated')[0]

# # visualize parcellation and ROI of interest. To see all labels names check the variable "label"
# Brain = mne.viz.get_brain_class()
# brain = Brain('fsaverage', 'split', 'inflated', subjects_dir=subjects_dir,
#           cortex='low_contrast', background='white', size=(800, 600))
# brain.add_label(label_L, borders=False)  
# brain.add_label(label_R, borders=False)  



# Extracting time course

Impl_Contra = list()
Impl_Ipsi = list()
Memo_Contra = list()
Memo_Ipsi = list()


for idx, val in enumerate(subjlist):
    
    this_Impl_Contra = list()
    this_Impl_Ipsi = list()
    this_Memo_Contra = list()
    this_Memo_Ipsi = list()
    
    this_Impl_Contra.append(Impl_LL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra.append(Impl_LR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Contra.append(Impl_RL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra.append(Impl_RR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Contra = np.mean(np.stack(this_Impl_Contra), axis = 0)
    Impl_Contra.append(this_Impl_Contra)
    
    this_Impl_Ipsi.append(Impl_LL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi.append(Impl_LR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Ipsi.append(Impl_RL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi.append(Impl_RR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Ipsi = np.mean(np.stack(this_Impl_Ipsi), axis = 0)
    Impl_Ipsi.append(this_Impl_Ipsi) 
    
    this_Memo_Contra.append(Memo_LL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra.append(Memo_LR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Contra.append(Memo_RL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra.append(Memo_RR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Contra = np.mean(np.stack(this_Memo_Contra), axis = 0)
    Memo_Contra.append(this_Memo_Contra)    

    this_Memo_Ipsi.append(Memo_LL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi.append(Memo_LR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Ipsi.append(Memo_RL[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi.append(Memo_RR[idx]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Ipsi = np.mean(np.stack(this_Memo_Ipsi), axis = 0)
    Memo_Ipsi.append(this_Memo_Ipsi) 




###############################################################################################################
# STATS
###############################################################################################################

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations

X = np.stack([Impl_Contra, Impl_Ipsi, Memo_Contra, Memo_Ipsi])
#X = np.squeeze(X)

X = X[:, :, :, -231:]

X_list = list(X)


# rmANOVA settings
# effect A is the one repeating slowest (here: Task)
# effect B is the one repeating faster (here: Laterality)
# interaction A:B

factor_levels = [2, 2]
effects_labels = ['Contra_Impl', 'Ipsi_Impl', 'Contra_Memo', 'Ipsi_Memo']
n_conditions = len(effects_labels)
n_subj = len(subjlist)
times = Impl_LL[0]['beta'].copy().crop(0, 1.8).times*1000
n_times = len(Impl_LL[0]['beta'].copy().crop(0, 1.8).times)


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

T_obs_A, clusters_A, cluster_p_values_A, h0 = permutation_cluster_test(
    X_list, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


## Main effect of RESP SIDE

effects = 'B'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

T_obs_B, clusters_B, cluster_p_values_B, h0 = permutation_cluster_test(
    X_list, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')



# P = 0.0023

# Computing effect size

for i_c, c in enumerate(clusters_B):
    c = c[0]
    if cluster_p_values_B[i_c] <= 0.05:
        idx = np.where(c == True)[0]
        
X1 = np.stack([Impl_Contra, Memo_Contra]).mean(axis = 0).squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)
X2 = np.stack([Impl_Ipsi, Memo_Ipsi]).mean(axis = 0).squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)

CohenD = np.mean(X1 - X2) / np.std(X1 - X2)
# d = 0.69

# plt.figure()
# plt.plot([X1, X2], 'o-')







## Interaction TASK * RESP SIDE

effects = 'A:B'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

T_obs_AB, clusters_AB, cluster_p_values_AB, h0 = permutation_cluster_test(
    X_list, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')







###############################################################################################################
# STATS
###############################################################################################################
from scipy import stats

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations

X_I = np.stack(Impl_Contra) - np.stack(Impl_Ipsi)
X_M = np.stack(Memo_Contra) - np.stack(Memo_Ipsi)

X = X_I - X_M
X = X[:, :, -231:]
X = X.squeeze()

pthresh = 0.05  # set threshold rather high to save some time
t_threshold = -stats.distributions.t.ppf(pthresh / 2., len(subjlist) - 1)

tail = -1 # 
n_permutations = 100000

T_obs_post, clusters_post, cluster_p_values_post, h0 = permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = -t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


# P = 0.07

# Computing effect size

for i_c, c in enumerate(clusters_post):
    c = c[0]
    if cluster_p_values_post[i_c] <= 0.08:
        idx = [c.start, c.stop]
        
X1 = X_I.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)
X2 = X_M.squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)

CohenD = np.mean(X1 - X2) / np.std(X1 - X2)
# d = 0.49

# plt.figure()
# plt.plot([X1, X2], 'o-')




###############################################################################################################
# PLOTS
###############################################################################################################


# Preparing dataframe for seaborn, long format
Impl_CvsI = np.stack(np.squeeze(Impl_Contra)) - np.stack(np.squeeze(Impl_Ipsi))
Memo_CvsI = np.stack(np.squeeze(Memo_Contra)) - np.stack(np.squeeze(Memo_Ipsi))

data = pd.DataFrame()
data['value'] = np.concatenate((np.squeeze(Impl_CvsI).flatten(), np.squeeze(Memo_CvsI).flatten()))
data['timepoint'] = np.tile(Impl_LL[0]['beta'].crop(-0.1, 1.8).times, len(subjlist)*2)*1000
data['Task'] = ['Implementation'] * Impl_CvsI.shape[1] * Impl_CvsI.shape[0] + ['Memorization'] * Memo_CvsI.shape[1] * Memo_CvsI.shape[0]


times_plot = Impl_LL[0]['beta'].crop(-0.1, 1.8).times
neg_timep = (times_plot < 0).sum()
    
# Plot with significance for main effect of Resp Side

fig, ax = plt.subplots()
sns.lineplot(x="timepoint", y="value",
             hue="Task", palette = ['blue', 'red'], ci = 68, 
             data=data)
plt.axhline(0, color = 'k', ls='--')
plt.axvline(0, color = 'k', ls='--')

for i_c, c in enumerate(clusters_B):
    c = c[0]
    idx = np.where(c == True)
    if cluster_p_values_B[i_c] <= 0.05:
        h = plt.axvspan(times[idx[0][0]] + neg_timep, times[idx[0][-1]] + neg_timep,
                        color='lightgrey', alpha=0.5)

for i_c, c in enumerate(clusters_post):
    c = c[0]
    # idx = np.where(c == True)
    if cluster_p_values_post[i_c] <= 0.08:
        h = plt.axvspan(times[c.start] + neg_timep, times[c.stop] + neg_timep,
                        color='darkgrey', alpha=0.5)
        
plt.xlabel('Time (ms)')
plt.ylabel('Power (au)')
plt.title('Contra vs Ipsi Hand ROIs')




###############################################################################################################
# SOURCE PLOT - just for visualization purposes
###############################################################################################################

Impl_Left_stc = list()
Impl_Right_stc = list()
Memo_Left_stc = list()
Memo_Right_stc = list()

for idx, val in enumerate(subjlist):
    
    this_Impl_Left = list()
    this_Impl_Right = list()
    this_Memo_Left = list()
    this_Memo_Right = list()
    
    this_Impl_Left.append(Impl_LL[idx]['beta'].crop(-0.1, 1.8).data)
    this_Impl_Left.append(Impl_RL[idx]['beta'].crop(-0.1, 1.8).data)  
    this_Impl_Left = np.mean(np.stack(this_Impl_Left), axis = 0)
    Impl_Left_stc.append(this_Impl_Left)

    this_Impl_Right.append(Impl_LR[idx]['beta'].crop(-0.1, 1.8).data)
    this_Impl_Right.append(Impl_RR[idx]['beta'].crop(-0.1, 1.8).data)  
    this_Impl_Right = np.mean(np.stack(this_Impl_Right), axis = 0)
    Impl_Right_stc.append(this_Impl_Right)

    this_Memo_Left.append(Memo_LL[idx]['beta'].crop(-0.1, 1.8).data)
    this_Memo_Left.append(Memo_RL[idx]['beta'].crop(-0.1, 1.8).data)  
    this_Memo_Left = np.mean(np.stack(this_Memo_Left), axis = 0)
    Memo_Left_stc.append(this_Memo_Left)

    this_Memo_Right.append(Memo_LR[idx]['beta'].crop(-0.1, 1.8).data)
    this_Memo_Right.append(Memo_RR[idx]['beta'].crop(-0.1, 1.8).data)  
    this_Memo_Right = np.mean(np.stack(this_Memo_Right), axis = 0)
    Memo_Right_stc.append(this_Memo_Right)


Impl_Left_stc = np.mean(np.stack(Impl_Left_stc), axis = 0)
Impl_Right_stc = np.mean(np.stack(Impl_Right_stc), axis = 0)
Memo_Left_stc = np.mean(np.stack(Memo_Left_stc), axis = 0)
Memo_Right_stc = np.mean(np.stack(Memo_Right_stc), axis = 0)


Impl_Left = Impl_LL[0]['beta'].copy()
Impl_Left.data = Impl_Left_stc

Impl_Right = Impl_RR[0]['beta'].copy()
Impl_Right.data = Impl_Right_stc

Memo_Left = Memo_LL[0]['beta'].copy()
Memo_Left.data = Memo_Left_stc

Memo_Right = Memo_RR[0]['beta'].copy()
Memo_Right.data = Memo_Right_stc


# Here I mirror the conditions LEFT
# so that the hemispheres are switched and correspond to resp RIGHT

# To make the following work I had to slightly manually adjust the directories
# in the path mne_data\MNE-fsaverage-data\ create the folder fsaverage_sym and
# copy the folder surf, mri and label from the mne_data\MNE-fsaverage-data\fsaverage

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


# Compute a morph-matrix mapping the right to the left hemisphere,
# and vice-versa.
morph = mne.compute_source_morph(Impl_Left, 'fsaverage', 'fsaverage',
                                 spacing=Impl_Left.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error')  # creating morph map

Impl_Left_xhemi = morph.apply(Impl_Left)



morph = mne.compute_source_morph(Memo_Left, 'fsaverage', 'fsaverage',
                                 spacing=Memo_Left.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error')  # creating morph map

Memo_Left_xhemi = morph.apply(Memo_Left)



# To make them comparable, I also need to morph of the RIGHT conditions to 
# fsaverage_sym

Impl_Right = mne.compute_source_morph(Impl_Right, subject_from = 'fsaverage', 
                                      subject_to = 'fsaverage', smooth=5,
                                warn=False, spacing=Memo_Left.vertices,
                                subjects_dir=subjects_dir).apply(Impl_Right)

Memo_Right = mne.compute_source_morph(Memo_Right, subject_from = 'fsaverage', 
                                      subject_to = 'fsaverage', smooth=5,
                                warn=False, spacing=Memo_Left.vertices,
                                subjects_dir=subjects_dir).apply(Memo_Right)



# Averaging Left (Mirrored) and Right

Impl_avg = Impl_Right.copy()
Impl_avg.data = np.mean((Impl_Left_xhemi.data, Impl_Right.data), axis = 0)

Memo_avg = Memo_Right.copy()
Memo_avg.data = np.mean((Memo_Left_xhemi.data, Memo_Right.data), axis = 0)

# Computing again cross-hemispheric morphs

morph = mne.compute_source_morph(Impl_avg, 'fsaverage', 'fsaverage',
                                 spacing=Impl_avg.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error')  # creating morph map

Impl_avg_xhemi = morph.apply(Impl_avg)

morph = mne.compute_source_morph(Memo_avg, 'fsaverage', 'fsaverage',
                                 spacing=Memo_avg.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error')  # creating morph map

Memo_avg_xhemi = morph.apply(Memo_avg)


# Subtract

diff_Impl = Impl_avg - Impl_avg_xhemi
diff_Memo = Memo_avg - Memo_avg_xhemi

diff = diff_Impl - diff_Memo



mne.viz.set_3d_backend("mayavi")

#from mayavi import mlab

#fig = mlab.figure(size=(300, 300))
brain1 = diff_Impl.plot(
    subject='fsaverage',
    hemi='lh',
    background='white',
    foreground='black',
    time_label='',
    initial_time=1,
    smoothing_steps=10,
    alpha = 1,
    title = 'Implementation',
    clim=dict(kind='value', pos_lims=[0.4, 0.5, 0.8]),
    views = 'lat')



# fig = mlab.figure(size=(300, 300))
brain2 = diff_Memo.plot(
    subject='fsaverage',
    hemi='lh',
    background='white',
    foreground='black',
    time_label='',
    initial_time=1,
    smoothing_steps=10,
    alpha = 1,
    title = 'Memorization',
    clim=dict(kind='value', pos_lims=[0.4, 0.5, 0.8]),
    views = 'lat')










###############################################################################################################
# TESTING FOR TASK ORDER
###############################################################################################################

# Figuring out order
META_all = np.zeros(len(subjlist))

data_path = base_path + '/ParticipantsData/'

for idx_s, val in enumerate(subjlist):

    os.chdir(data_path + '/Subj' + str(val) + '/')
    meta = mne.read_epochs('Subj' + str(val) + '-epo.fif', 
                    verbose = False, preload = False).metadata
    if meta.reset_index().loc[0, 'Task'] == 'Impl':
        META_all[idx_s] = 1
    else:
        META_all[idx_s] = 2
       

# from itertools import compress
# subjlist_IM = list(compress(subjlist, META_all == 1))
# subjlist_MI = list(compress(subjlist, META_all == 2))

subjlist_IM = [i for i, j in enumerate(subjlist) if META_all[i] == 1]
subjlist_MI = [i for i, j in enumerate(subjlist) if META_all[i] == 2]

# Extracting time course

Impl_Contra_IM = list()
Impl_Ipsi_IM = list()
Memo_Contra_IM = list()
Memo_Ipsi_IM = list()


for idx, val in enumerate(subjlist_IM):
    
    this_Impl_Contra_IM = list()
    this_Impl_Ipsi_IM = list()
    this_Memo_Contra_IM = list()
    this_Memo_Ipsi_IM = list()
    
    this_Impl_Contra_IM.append(Impl_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra_IM.append(Impl_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Contra_IM.append(Impl_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra_IM.append(Impl_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Contra_IM = np.mean(np.stack(this_Impl_Contra_IM), axis = 0)
    Impl_Contra_IM.append(this_Impl_Contra_IM)
    
    this_Impl_Ipsi_IM.append(Impl_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi_IM.append(Impl_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Ipsi_IM.append(Impl_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi_IM.append(Impl_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Ipsi_IM = np.mean(np.stack(this_Impl_Ipsi_IM), axis = 0)
    Impl_Ipsi_IM.append(this_Impl_Ipsi_IM) 
    
    this_Memo_Contra_IM.append(Memo_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra_IM.append(Memo_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Contra_IM.append(Memo_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra_IM.append(Memo_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Contra_IM = np.mean(np.stack(this_Memo_Contra_IM), axis = 0)
    Memo_Contra_IM.append(this_Memo_Contra_IM)    

    this_Memo_Ipsi_IM.append(Memo_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi_IM.append(Memo_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Ipsi_IM.append(Memo_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi_IM.append(Memo_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Ipsi_IM = np.mean(np.stack(this_Memo_Ipsi_IM), axis = 0)
    Memo_Ipsi_IM.append(this_Memo_Ipsi_IM) 
    
    
    

  # Extracting time course

Impl_Contra_MI = list()
Impl_Ipsi_MI = list()
Memo_Contra_MI = list()
Memo_Ipsi_MI = list()


for idx, val in enumerate(subjlist_MI):
    
    this_Impl_Contra_MI = list()
    this_Impl_Ipsi_MI = list()
    this_Memo_Contra_MI = list()
    this_Memo_Ipsi_MI = list()
    
    this_Impl_Contra_MI.append(Impl_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra_MI.append(Impl_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Contra_MI.append(Impl_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra_MI.append(Impl_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Contra_MI = np.mean(np.stack(this_Impl_Contra_MI), axis = 0)
    Impl_Contra_MI.append(this_Impl_Contra_MI)
    
    this_Impl_Ipsi_MI.append(Impl_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi_MI.append(Impl_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Ipsi_MI.append(Impl_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi_MI.append(Impl_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Ipsi_MI = np.mean(np.stack(this_Impl_Ipsi_MI), axis = 0)
    Impl_Ipsi_MI.append(this_Impl_Ipsi_MI) 
    
    this_Memo_Contra_MI.append(Memo_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra_MI.append(Memo_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Contra_MI.append(Memo_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra_MI.append(Memo_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Contra_MI = np.mean(np.stack(this_Memo_Contra_MI), axis = 0)
    Memo_Contra_MI.append(this_Memo_Contra_MI)    

    this_Memo_Ipsi_MI.append(Memo_LL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi_MI.append(Memo_LR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Ipsi_MI.append(Memo_RL[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi_MI.append(Memo_RR[val]['beta'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Ipsi_MI = np.mean(np.stack(this_Memo_Ipsi_MI), axis = 0)
    Memo_Ipsi_MI.append(this_Memo_Ipsi_MI) 
      



###############################################################################################################
# STATS
###############################################################################################################

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations

# testing for the main effect of Task Order

X1 = np.stack([Impl_Contra_IM, Impl_Ipsi_IM, Memo_Contra_IM, Memo_Ipsi_IM])
X1 = np.squeeze(X1)

X2 = np.stack([Impl_Contra_MI, Impl_Ipsi_MI, Memo_Contra_MI, Memo_Ipsi_MI])
X2 = np.squeeze(X2)

X1 = X1[:, :, :, -231:]
X2 = X2[:, :, :, -231:]

#reshape so that pp is the first dimension
X1 = np.transpose(X1, [1, 0, 2])
X2 = np.transpose(X2, [1, 0, 2])

X_list = list([X1, X2])


n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

T_obs_A, clusters_A, cluster_p_values_A, h0 = permutation_cluster_test(
    X_list, stat_fun=None, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')

## NO clusters!


# Here I test for the three way interaction Task Order x Task x Laterality

X_I_IM = np.stack(Impl_Contra_IM) - np.stack(Impl_Ipsi_IM)
X_M_IM = np.stack(Memo_Contra_IM) - np.stack(Memo_Ipsi_IM)
X_I_MI = np.stack(Impl_Contra_MI) - np.stack(Impl_Ipsi_MI)
X_M_MI = np.stack(Memo_Contra_MI) - np.stack(Memo_Ipsi_MI)

X_I_IM = np.squeeze(X_I_IM)
X_M_IM = np.squeeze(X_M_IM)
X_I_MI = np.squeeze(X_I_MI)
X_M_MI = np.squeeze(X_M_MI)


X_I_IM = X_I_IM[:,  -231:]
X_M_IM = X_M_IM[:,  -231:]
X_I_MI = X_I_MI[:,  -231:]
X_M_MI = X_M_MI[:,  -231:]


X_list = list([X_I_IM - X_M_IM, X_I_MI - X_M_MI])


n_permutations = 10000  # Save some time (the test won't be too sensitive ...)

T_obs_A, clusters_A, cluster_p_values_A, h0 = permutation_cluster_test(
    X_list, stat_fun=None, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


## One cluster p = 0.74


## PLOT

# Preparing dataframe for seaborn, long format
Impl_CvsI_IM = np.stack(np.squeeze(Impl_Contra_IM)) - np.stack(np.squeeze(Impl_Ipsi_IM))
Memo_CvsI_IM = np.stack(np.squeeze(Memo_Contra_IM)) - np.stack(np.squeeze(Memo_Ipsi_IM))
Impl_CvsI_MI= np.stack(np.squeeze(Impl_Contra_MI)) - np.stack(np.squeeze(Impl_Ipsi_MI))
Memo_CvsI_MI = np.stack(np.squeeze(Memo_Contra_MI)) - np.stack(np.squeeze(Memo_Ipsi_MI))


data = pd.DataFrame()
data['value'] = np.concatenate((np.squeeze(Impl_CvsI_IM).flatten(), np.squeeze(Memo_CvsI_IM).flatten(), np.squeeze(Impl_CvsI_MI).flatten(), np.squeeze(Memo_CvsI_MI).flatten()))

data['timepoint'] = np.concatenate((np.tile(Impl_LL[0]['beta'].crop(-0.1, 1.8).times, len(subjlist_IM)*2)*1000,
np.tile(Impl_LL[0]['beta'].crop(-0.1, 1.8).times, len(subjlist_MI)*2)*1000))

data['Task'] = (['Implementation'] * Impl_CvsI_IM.shape[1] * Impl_CvsI_IM.shape[0] + 
                ['Memorization'] * Memo_CvsI_IM.shape[1] * Memo_CvsI_IM.shape[0] +
                ['Implementation'] * Impl_CvsI_MI.shape[1] * Impl_CvsI_MI.shape[0] +
                ['Memorization'] * Memo_CvsI_MI.shape[1] * Memo_CvsI_MI.shape[0])

data['Task Order'] = (['Impl > Memo'] * Impl_CvsI_IM.shape[1] * Impl_CvsI_IM.shape[0] + 
                ['Impl > Memo'] * Memo_CvsI_IM.shape[1] * Memo_CvsI_IM.shape[0] +
                ['Memo > Impl'] * Impl_CvsI_MI.shape[1] * Impl_CvsI_MI.shape[0] +
                ['Memo > Impl'] * Memo_CvsI_MI.shape[1] * Memo_CvsI_MI.shape[0])



times_plot = Impl_LL[0]['beta'].crop(-0.1, 1.8).times
neg_timep = (times_plot < 0).sum()
    
# Plot with significance for main effect of Resp Side

fig, ax = plt.subplots()
sns.lineplot(x="timepoint", y="value",
             hue="Task", style = 'Task Order', palette = ['blue', 'red'], ci = 68, 
             data=data)
plt.axhline(0, color = 'k', ls='--')
plt.axvline(0, color = 'k', ls='--')











