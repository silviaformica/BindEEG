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
from mne.stats import permutation_cluster_test, spatio_temporal_cluster_test, summarize_clusters_stc
from mne.stats import f_mway_rm, f_threshold_mway_rm
import pickle 

base_path = 'E:/BindEEG/'
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

os.chdir(base_path + '/Sources/alpha_whole_brain')

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
    Impl_LL.append({'alpha' : mne.read_source_estimate('Impl_LL_%s' % val)})
    Impl_LR.append({'alpha' : mne.read_source_estimate('Impl_LR_%s' % val)})
    Impl_RL.append({'alpha' : mne.read_source_estimate('Impl_RL_%s' % val)})
    Impl_RR.append({'alpha' : mne.read_source_estimate('Impl_RR_%s' % val)})
    Memo_LL.append({'alpha' : mne.read_source_estimate('Memo_LL_%s' % val)})
    Memo_LR.append({'alpha' : mne.read_source_estimate('Memo_LR_%s' % val)})
    Memo_RL.append({'alpha' : mne.read_source_estimate('Memo_RL_%s' % val)})
    Memo_RR.append({'alpha' : mne.read_source_estimate('Memo_RR_%s' % val)})

    
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

# visualize parcellation and ROI of interest. To see all labels names check the variable "label"
Brain = mne.viz.get_brain_class()
brain = Brain('fsaverage', 'both', 'inflated', subjects_dir=subjects_dir,
          cortex='low_contrast', background='white', size=(800, 600))
brain.add_annotation('aparc')
label_to_plot = [label for label in labels if label.name == 'lateraloccipital-lh'][0]
brain.add_label(label_to_plot, borders=False)  
label_to_plot = [label for label in labels if label.name == 'lateraloccipital-rh'][0]
brain.add_label(label_to_plot, borders=False)  

# Choose ROI - bilaterally
label_L = [label for label in labels if label.name == 'lateraloccipital-lh'][0]
label_R = [label for label in labels if label.name == 'lateraloccipital-rh'][0]



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
    
    this_Impl_Contra.append(Impl_LL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra.append(Impl_LR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Contra.append(Impl_RL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Contra.append(Impl_RR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))        
    this_Impl_Contra = np.mean(np.stack(this_Impl_Contra), axis = 0)
    Impl_Contra.append(this_Impl_Contra)

    this_Impl_Ipsi.append(Impl_LL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi.append(Impl_LR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Impl_Ipsi.append(Impl_RL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Impl_Ipsi.append(Impl_RR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))        
    this_Impl_Ipsi = np.mean(np.stack(this_Impl_Ipsi), axis = 0)
    Impl_Ipsi.append(this_Impl_Ipsi)    

    this_Memo_Contra.append(Memo_LL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra.append(Memo_LR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Contra.append(Memo_RL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Contra.append(Memo_RR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))        
    this_Memo_Contra = np.mean(np.stack(this_Memo_Contra), axis = 0)
    Memo_Contra.append(this_Memo_Contra)    

    this_Memo_Ipsi.append(Memo_LL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi.append(Memo_LR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_L, src, mode = 'pca_flip'))
    this_Memo_Ipsi.append(Memo_RL[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))
    this_Memo_Ipsi.append(Memo_RR[idx]['alpha'].crop(-0.1, 1.8).extract_label_time_course(label_R, src, mode = 'pca_flip'))        
    this_Memo_Ipsi = np.mean(np.stack(this_Memo_Ipsi), axis = 0)
    Memo_Ipsi.append(this_Memo_Ipsi) 


###############################################################################################################
# STATS
###############################################################################################################

# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations

X = np.stack([Impl_Contra, Impl_Ipsi, Memo_Contra, Memo_Ipsi])
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
times = Impl_LL[0]['alpha'].copy().crop(0, 1.8).times*1000
n_times = len(Impl_LL[0]['alpha'].copy().crop(0, 1.8).times)


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

    ## NO cluster!


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


# P = 0.0002

# Computing effect size

for i_c, c in enumerate(clusters_B):
    c = c[0]
    if cluster_p_values_B[i_c] <= 0.05:
        idx = np.where(c == True)[0]
        
X1 = np.stack([Impl_Contra, Memo_Contra]).mean(axis = 0).squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)
X2 = np.stack([Impl_Ipsi, Memo_Ipsi]).mean(axis = 0).squeeze()[:, -231:][:, idx[0]:idx[-1]].mean(axis = 1)

CohenD = np.mean(X1 - X2) / np.std(X1 - X2)
# d = 1.3683873

# plt.figure()
# plt.plot([X1, X2], 'o-')


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

T_obs_AB, clusters_AB, cluster_p_values_AB, h0 = permutation_cluster_test(
    X_list, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=1,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')

    ## No cluster


###############################################################################################################
# PLOTS
###############################################################################################################

# Preparing dataframe for seaborn, long format
Impl_CvsI = np.stack(np.squeeze(Impl_Contra)) - np.stack(np.squeeze(Impl_Ipsi))
Memo_CvsI = np.stack(np.squeeze(Memo_Contra)) - np.stack(np.squeeze(Memo_Ipsi))

data = pd.DataFrame()
data['value'] = np.concatenate((np.squeeze(Impl_CvsI).flatten(), np.squeeze(Memo_CvsI).flatten()))
data['timepoint'] = np.tile(Impl_LL[0]['alpha'].crop(-0.1, 1.8).times, len(subjlist)*2)*1000
data['Task'] = ['Implementation'] * Impl_CvsI.shape[1] * Impl_CvsI.shape[0] + ['Memorization'] * Memo_CvsI.shape[1] * Memo_CvsI.shape[0]

times_plot = Impl_LL[0]['alpha'].crop(-0.1, 1.8).times
neg_timep = (times_plot < 0).sum()
    
# Plot with significance for main effect of Cued Side

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
plt.xlabel('Time (ms)')
plt.ylabel('Power (au)')
plt.title('Contra vs Ipsi Lateral-Occipital ROI')


###############################################################################################################
# OTHER PLOTS
###############################################################################################################

# All lines separately

# data = pd.DataFrame()
# data['value'] = np.concatenate((np.stack(Impl_Contra).flatten(), np.stack(Impl_Ipsi).flatten(), np.stack(Memo_Contra).flatten(), np.stack(Memo_Ipsi).flatten()))
# data['timepoint'] = np.tile(Impl_LL[0]['beta'].crop(-0.1, 1.8).times, len(subjlist)*4)*1000
# data['Task'] = ['Implementation'] * int(len(data)/2) +    ['Memorization'] * int(len(data)/2)
# data['Laterality'] = ['Contra'] * int(len(data)/4) + ['Ipsi'] * int(len(data)/4) + ['Contra'] * int(len(data)/4) + ['Ipsi'] * int(len(data)/4)


# fig, ax = plt.subplots()
# sns.lineplot(x="timepoint", y="value",
#               hue="Task", style = 'Laterality', palette = ['blue', 'red'], ci = 68, 
#               data=data)
# plt.axhline(0, color = 'k', ls='--')
# plt.axvline(0, color = 'k', ls='--')

# # for i_c, c in enumerate(clusters_AB):
# #     c = c[0]
# #     if cluster_p_values_AB[i_c] <= 0.1:
# #         h = plt.axvspan(times[c.start], times[c.stop - 1],
# #                         color='lightgrey', alpha=0.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Power')
# plt.title('Contra vs Ipsi superior PC')


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
    
    this_Impl_Left.append(Impl_LL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Impl_Left.append(Impl_LR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Impl_Left = np.mean(np.stack(this_Impl_Left), axis = 0)
    Impl_Left_stc.append(this_Impl_Left)

    this_Impl_Right.append(Impl_RL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Impl_Right.append(Impl_RR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Impl_Right = np.mean(np.stack(this_Impl_Right), axis = 0)
    Impl_Right_stc.append(this_Impl_Right)

    this_Memo_Left.append(Memo_LL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Memo_Left.append(Memo_LR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Memo_Left = np.mean(np.stack(this_Memo_Left), axis = 0)
    Memo_Left_stc.append(this_Memo_Left)

    this_Memo_Right.append(Memo_RL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Memo_Right.append(Memo_RR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Memo_Right = np.mean(np.stack(this_Memo_Right), axis = 0)
    Memo_Right_stc.append(this_Memo_Right)


Impl_Left_stc = np.mean(np.stack(Impl_Left_stc), axis = 0)
Impl_Right_stc = np.mean(np.stack(Impl_Right_stc), axis = 0)
Memo_Left_stc = np.mean(np.stack(Memo_Left_stc), axis = 0)
Memo_Right_stc = np.mean(np.stack(Memo_Right_stc), axis = 0)


Impl_Left = Impl_LL[0]['alpha'].copy()
Impl_Left.data = Impl_Left_stc

Impl_Right = Impl_RR[0]['alpha'].copy()
Impl_Right.data = Impl_Right_stc

Memo_Left = Memo_LL[0]['alpha'].copy()
Memo_Left.data = Memo_Left_stc

Memo_Right = Memo_RR[0]['alpha'].copy()
Memo_Right.data = Memo_Right_stc


# Here I mirror the conditions LEFT
# so that the hemispheres are switched and correspond to orient RIGHT

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

mne.viz.set_3d_backend("pyvista")


diff_Impl.plot(hemi='both', subjects_dir=subjects_dir,
               background = 'white', 
          size=(800, 600))


diff_Memo.plot(hemi='both', subjects_dir=subjects_dir,
               background = 'white', 
          size=(800, 600))



mne.viz.set_3d_backend("mayavi")

#from mayavi import mlab

#fig = mlab.figure(size=(300, 300))
brain1 = diff_Impl.plot(
    subject='fsaverage',
    hemi='lh',
    background='white',
    foreground='black',
    time_label='',
    initial_time=0.7,
    smoothing_steps=10,
    alpha = 1,
    title = 'Implementation',
    clim=dict(kind='value', pos_lims=[0.9, 1, 3.2]),
    views = 'lat')





# fig = mlab.figure(size=(300, 300))
brain2 = diff_Memo.plot(
    subject='fsaverage',
    hemi='lh',
    background='white',
    foreground='black',
    time_label='',
    initial_time=0.7,
    smoothing_steps=10,
    alpha = 1,
    title = 'Memorization',
    clim=dict(kind='value', pos_lims=[0.9, 1, 3.2]),
    views = 'lat')









###############################################################################################################
# STATS on sources
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
    
    this_Impl_Left.append(Impl_LL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Impl_Left.append(Impl_LR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Impl_Left = np.mean(np.stack(this_Impl_Left), axis = 0)
    Impl_Left_stc.append(this_Impl_Left)

    this_Impl_Right.append(Impl_RL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Impl_Right.append(Impl_RR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Impl_Right = np.mean(np.stack(this_Impl_Right), axis = 0)
    Impl_Right_stc.append(this_Impl_Right)

    this_Memo_Left.append(Memo_LL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Memo_Left.append(Memo_LR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Memo_Left = np.mean(np.stack(this_Memo_Left), axis = 0)
    Memo_Left_stc.append(this_Memo_Left)

    this_Memo_Right.append(Memo_RL[idx]['alpha'].crop(-0.1, 1.8).data)
    this_Memo_Right.append(Memo_RR[idx]['alpha'].crop(-0.1, 1.8).data)  
    this_Memo_Right = np.mean(np.stack(this_Memo_Right), axis = 0)
    Memo_Right_stc.append(this_Memo_Right)


Impl_avg = list()
Impl_avg_xhemi = list()
Memo_avg = list()
Memo_avg_xhemi = list()

    
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

for idx, val in enumerate(subjlist):
    
    Impl_Left = Impl_LL[idx]['alpha'].copy()
    Impl_Left.data = Impl_Left_stc[idx]
    
    Impl_Right = Impl_RR[idx]['alpha'].copy()
    Impl_Right.data = Impl_Right_stc[idx]
    
    Memo_Left = Memo_LL[idx]['alpha'].copy()
    Memo_Left.data = Memo_Left_stc[idx]
    
    Memo_Right = Memo_RR[idx]['alpha'].copy()
    Memo_Right.data = Memo_Right_stc[idx]
    

    
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
    
    

    Impl_Right = mne.compute_source_morph(Impl_Right, subject_from = 'fsaverage', 
                                         subject_to = 'fsaverage', smooth=5,
                                   warn=False, spacing=Impl_Right.vertices,
                                   subjects_dir=subjects_dir).apply(Impl_Right)
    
    Memo_Right = mne.compute_source_morph(Memo_Right, subject_from = 'fsaverage', 
                                         subject_to = 'fsaverage', smooth=5,
                                   warn=False, spacing=Memo_Right.vertices,
                                   subjects_dir=subjects_dir).apply(Memo_Right)



    
    # Averaging Left (Mirrored) and Right
    
    this_Impl_avg = Impl_Right.copy()
    this_Impl_avg.data = np.mean((Impl_Left_xhemi.data, Impl_Right.data), axis = 0)
    
    Impl_avg.append(this_Impl_avg)
    
    this_Memo_avg = Memo_Right.copy()
    this_Memo_avg.data = np.mean((Memo_Left_xhemi.data, Memo_Right.data), axis = 0)
    
    Memo_avg.append(this_Memo_avg)
    
    # Computing again cross-hemispheric morphs
    
    morph = mne.compute_source_morph(this_Impl_avg, 'fsaverage', 'fsaverage',
                                     spacing=this_Impl_avg.vertices, warn=False,
                                     subjects_dir=subjects_dir, xhemi=True,
                                     verbose='error')  # creating morph map
    
    Impl_avg_xhemi.append(morph.apply(this_Impl_avg))
    
    morph = mne.compute_source_morph(this_Memo_avg, 'fsaverage', 'fsaverage',
                                     spacing=this_Memo_avg.vertices, warn=False,
                                     subjects_dir=subjects_dir, xhemi=True,
                                     verbose='error')  # creating morph map
    
    Memo_avg_xhemi.append(morph.apply(this_Memo_avg))




    
    
sum(Memo_avg_xhemi).plot(hemi='both', subjects_dir=subjects_dir,
          size=(800, 600))
       
sum(Memo_avg).plot(hemi='both', subjects_dir=subjects_dir,
          size=(800, 600))
   
(sum(Memo_avg) - sum(Memo_avg_xhemi)).plot(hemi='both', subjects_dir=subjects_dir,
          size=(800, 600))
   
(sum(Impl_avg) - sum(Impl_avg_xhemi)).plot(hemi='both', subjects_dir=subjects_dir,
          size=(800, 600))
   




# Extract data from only left hemi & make list for clustering
# _avg corresponds to a condition in which the left hemisphere is Contra and avg_xhemi is Ipsi

Impl_Contra = list()
Impl_Ipsi = list()
Memo_Contra = list()
Memo_Ipsi = list()

# for idx, val in enumerate(subjlist):
#     Impl_Contra.append(Impl_avg[idx].lh_data)
#     Impl_Ipsi.append(Impl_avg_xhemi[idx].lh_data)
#     Memo_Contra.append(Memo_avg[idx].lh_data)
#     Memo_Ipsi.append(Memo_avg_xhemi[idx].lh_data)
    
    
for idx, val in enumerate(subjlist):
   Diff_Impl = Impl_avg[idx] - Impl_avg_xhemi[idx]
   Diff_Memo = Memo_avg[idx] - Memo_avg_xhemi[idx]
   Impl_Contra.append(Diff_Impl.lh_data)
   Impl_Ipsi.append(Diff_Impl.rh_data)
   Memo_Contra.append(Diff_Memo.lh_data)
   Memo_Ipsi.append(Diff_Memo.rh_data)
       
    
    
    

    
## Deleting some variables
del(Impl_avg, Impl_avg_xhemi, Memo_avg, Memo_avg_xhemi, this_Memo_avg, this_Impl_avg)   
del(Impl_Left_stc, Impl_Right_stc, Memo_Left_stc, Memo_Right_stc) 
del(this_Impl_Left, this_Impl_Right, this_Memo_Left, this_Memo_Right)   
del(Impl_Left, Impl_Right, Memo_Left, Memo_Right)
del(Memo_Left_xhemi, Impl_Left_xhemi)
   
###############################################################################################################
# STATS on SOURCES - left HEMISPHERE only!!!
###############################################################################################################


src_name = op.join(fs_dir, 'bem', 'fsaverage-oct-6-src.fif')
src = mne.read_source_spaces(src_name, verbose = True)

# as we only have one hemisphere we need only need half the adjacency
print('Computing adjacency.')
adjacency = mne.spatial_src_adjacency(src[:1])

fig = plt.figure()
ax = fig.add_subplot(111, facecolor='black')
ax.plot(adjacency.col, adjacency.row, 's', color='white', ms=1)
ax.set_xlim(0, adjacency.shape[1])
ax.set_ylim(0, adjacency.shape[0])
ax.set_aspect('equal')
for spine in ax.spines.values():
    spine.set_visible(False)
ax.invert_yaxis()
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])



# Data: for the function permutation_cluster_test() it needs to be a list of arrays
# each array contains data for one group/condition of observations
X = np.stack([Impl_Contra, Impl_Ipsi, Memo_Contra, Memo_Ipsi])
X = np.transpose(X, [0, 1, 3, 2]) # conditions x subj x time x space
X_list = list(X)


# rmANOVA settings
# effect A is the one repeating slowest (here: Task)
# effect B is the one repeating faster (here: Laterality)
# interaction A:B

factor_levels = [2, 2]
effects_labels = ['Contra_Impl', 'Ipsi_Impl', 'Contra_Memo', 'Ipsi_Memo']
n_conditions = len(effects_labels)
n_subj = len(subjlist)
times = Impl_LL[0]['alpha'].crop(-0.1, 1.8).times*1000
n_times = len(Impl_LL[0]['alpha'].crop(-0.1, 1.8).times)


# Main effect of TASK

effects = 'A'
def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]      #don't need p_values here

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_A = \
    spatio_temporal_cluster_test(X, adjacency=adjacency, n_jobs=4,
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
pthresh = 0.00001  # set threshold rather high to save some time
f_thresh = f_threshold_mway_rm(n_subj, factor_levels, effects, pthresh)

tail = 1  # f-test, so tail > 0
n_permutations = 100  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_B = \
    spatio_temporal_cluster_test(X, adjacency=adjacency, n_jobs=4,
                                  threshold=f_thresh, stat_fun=stat_fun,
                                  n_permutations=n_permutations)


with open('clu_B.pickle', 'wb') as f:
    pickle.dump(clu_B, f)

del(clu_B)

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
    spatio_temporal_cluster_test(X, adjacency=adjacency, n_jobs=2,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations)



with open('clu_AB.pickle', 'wb') as f:
    pickle.dump(clu_AB, f)









###############################################################################################################
# PLOTTING CLUSTERS
###############################################################################################################

# Loading Clusters

with open('clu_B.pickle', 'rb') as f:
      clu_B = pickle.load(f)


good_cluster_inds = np.where(clu_B[2] < 0.05)[0]



print('Visualizing clusters.')
fsave_vertices = [src[0]['vertno'], []]
tstep = Impl_LL[0]['alpha'].tstep * 1000
#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu_B, tstep=tstep, p_thresh = 0.05,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration

# The brighter the color, the stronger the interaction between
# stimulus modality and stimulus location

mne.viz.set_3d_backend("pyvista")

brain = stc_all_cluster_vis.plot(subject='fsaverage', views='lat',
                                 time_label='temporal extent (ms)',
                                 cortex='low_contrast', transparent = True,
                                 clim=dict(kind='value', lims=[0, 100, 500]))





prova = np.stack(Impl_Contra)
prova = prova.transpose([0, 2, 1])

from scipy import stats
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)
n_permutations = 100

T_obs, clusters, cluster_p_values, H0 = clu_AB = \
    mne.stats.permutation_cluster_1samp_test(prova, adjacency=adjacency, n_jobs=4,
                                 threshold=t_threshold, 
                                 n_permutations=n_permutations)



print('Visualizing clusters.')
fsave_vertices = [src[0]['vertno'], []]
tstep = Impl_LL[0]['alpha'].tstep * 1000
#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu_AB, tstep=tstep, p_thresh = 0.05,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration

# The brighter the color, the stronger the interaction between
# stimulus modality and stimulus location

mne.viz.set_3d_backend("pyvista")

brain = stc_all_cluster_vis.plot(subject='fsaverage', views='lat',
                                 time_label='temporal extent (ms)',
                                 cortex='low_contrast', transparent = False)

