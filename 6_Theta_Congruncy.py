# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:19:01 2021

@author: Carlos
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

subjlist = [ 1, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

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

dist = 0.02    # need to find the right threshold!

from scipy import sparse
from scipy.spatial.distance import cdist
adj = cdist(src[0]['rr'][src[0]['vertno']],
            src[1]['rr'][src[1]['vertno']])
adj = sparse.csr_matrix(adj <= dist, dtype=int)
empties = [sparse.csr_matrix((nv, nv), dtype=int) for nv in adj.shape]
adj = sparse.vstack([sparse.hstack([empties[0], adj]),
                     sparse.hstack([adj.T, empties[1]])])

adj_tot = sparse.coo_matrix(adjacency + adj)


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
I_contr = np.transpose(I_contr, [0, 2, 1])

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_I = \
    spatio_temporal_cluster_1samp_test(I_contr, adjacency=adj_tot, n_jobs=4, n_permutations = n_permutations,
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
M_contr = np.transpose(M_contr, [0, 2, 1])

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_M = \
    spatio_temporal_cluster_1samp_test(M_contr, adjacency=adj_tot, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)




###############################################################################################################
# Between Tasks
###############################################################################################################

Diff = I_contr - M_contr
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_IM = \
    spatio_temporal_cluster_1samp_test(Diff, adjacency=adj_tot, n_jobs=4, n_permutations = n_permutations,
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
Diff = np.transpose(Diff, [0, 2, 1])

p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_IvsM_congr = \
    spatio_temporal_cluster_1samp_test(Diff, adjacency=adj_tot, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)


###############################################################################################################
# Impl vs Memo only in incongruent trials
###############################################################################################################


I_incongr = np.stack([Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft])
I_incongr = np.mean(I_incongr, axis = 0)
           
M_incongr = np.stack([Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft])
M_incongr = np.mean(M_incongr, axis = 0)

Diff = I_incongr - M_incongr
Diff = np.transpose(Diff, [0, 2, 1])

p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_IvsM_incongr = \
    spatio_temporal_cluster_1samp_test(Diff, adjacency=adj_tot, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)



###############################################################################################################
# Congruent vs INcongruent across tasks
###############################################################################################################


congr = np.stack([Impl_CuedLeft_RespLeft, Impl_CuedRight_RespRight, Memo_CuedLeft_RespLeft, Memo_CuedRight_RespRight])
congr = np.mean(congr, axis = 0)
           
incongr = np.stack([Impl_CuedLeft_RespRight, Impl_CuedRight_RespLeft, Memo_CuedLeft_RespRight, Memo_CuedRight_RespLeft])
incongr = np.mean(incongr, axis = 0)

Diff = incongr - congr
Diff = np.transpose(Diff, [0, 2, 1])

p_threshold = 0.001
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1  # f-test, so tail > 0
n_permutations = 500  # Save some time (the test won't be too sensitive ...)

print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu_congr_vs_incongr = \
    spatio_temporal_cluster_1samp_test(Diff, adjacency=adj_tot, n_jobs=4, n_permutations = n_permutations,
                                       tail = tail,
                                       threshold= t_threshold, buffer_size=None,
                                       verbose=True)


###############################################################################################################
# saving
###############################################################################################################

with open('clu_I.pickle', 'wb') as f:
    pickle.dump(clu_I, f)

with open('clu_M.pickle', 'wb') as f:
    pickle.dump(clu_M, f)
    
with open('clu_IM.pickle', 'wb') as f:
    pickle.dump(clu_IM, f)
    
with open('clu_IvsM_congr.pickle', 'wb') as f:
    pickle.dump(clu_IvsM_congr, f)
    
with open('clu_IvsM_incongr.pickle', 'wb') as f:
    pickle.dump(clu_IvsM_incongr, f)
    
with open('clu_congr_vs_incongr.pickle', 'wb') as f:
    pickle.dump(clu_congr_vs_incongr, f)

    

###############################################################################################################
# Checking significance and plotting
###############################################################################################################



# Loading Clusters

with open('clu_I.pickle', 'rb') as f:
      clu_I = pickle.load(f)

with open('clu_M.pickle', 'rb') as f:
      clu_M = pickle.load(f)
      
with open('clu_IM.pickle', 'rb') as f:
      clu_IM = pickle.load(f)

with open('clu_IvsM_congr.pickle', 'rb') as f:
      clu_IvsM_congr = pickle.load(f)

with open('clu_IvsM_incongr.pickle', 'rb') as f:
      clu_IvsM_incongr = pickle.load(f)

with open('clu_congr_vs_incongr.pickle', 'rb') as f:
      clu_congr_vs_incongr = pickle.load(f)


np.sort(clu_I[2])
np.sort(clu_M[2])
np.sort(clu_IM[2])
np.sort(clu_IvsM_congr[2])
np.sort(clu_IvsM_incongr[2])
np.sort(clu_congr_vs_incongr[2])




print('Visualizing clusters.')
fsave_vertices = src
tstep = 7.8125


stc_all_cluster_vis = summarize_clusters_stc(clu_IvsM_congr, tstep=tstep, p_thresh=0.05,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration

mne.viz.set_3d_backend("pyvista")

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
time_points_clu = np.unique(clu_IvsM_congr[1][6][0])

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





























