# -*- coding: utf-8 -*-

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
# LatOcc ROIs
###############################################################################################################


labels = mne.read_labels_from_annot(
'fsaverage', 'aparc', 'both', subjects_dir=subjects_dir)    


# Choose ROI - bilaterally
label_L = [label for label in labels if label.name == 'lateraloccipital-lh'][0]
label_R = [label for label in labels if label.name == 'lateraloccipital-rh'][0]

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
plt.title('Implementation vs Memorization - LatOcc ROI')






###############################################################################################################
# Hand ROIs
###############################################################################################################

from scipy import spatial

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
plt.title('Implementation vs Memorization - Hand ROI')
