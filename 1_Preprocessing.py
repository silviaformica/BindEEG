# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:44:00 2020

@author: Silvia Formica silvia.formica@ugent.be

PREPROCESSING STEPS for the experiment BIND EEG
Set participant number at the beginning of the script

"""

###############################################################################################################
## IMPORTING MODULES
###############################################################################################################

import mne
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as op

###############################################################################################################
## SET DATA PATHS
###############################################################################################################

subj = 5

base_path = 'D:/BindEEG/ParticipantsData/'
data_path = base_path + 'Subj' + str(subj) + '/'
os.chdir(data_path)

###############################################################################################################
## IMPORTING RAW DATA
###############################################################################################################

raw = mne.io.read_raw_bdf(data_path + 'Subj' + str(subj) + '.bdf', preload=True)
print(raw.info)

###############################################################################################################
## MONTAGE
###############################################################################################################

# We need to specify the channel types in our recordings. The first 64 are the regular EEG channels
# EXG1 - EXG2 = left and right mastoids (as miscto differentiate from the others)
# EXG3 - EXG4 = left and right HEOG outer canthi (as emg)
# EXG5 - EXG6 = up and down VEOG above and belowleft eye (as emg)
# EXG7 - EXG8 = empty (as emg)
# Status = triggers channel (needs to be set to 'stim')

# create a list of 'EEG' channels
types = ['eeg']*73
# change elements in that list that aren't 'EEG' channels
types[-1] = 'stim'; types[-2] = 'emg'; types[-3] = 'emg'; types[-4] = 'emg'; types[-5] = 'emg';
types[-6] = 'emg'; types[-7] = 'emg'; types[-8] = 'misc'; types[-9] = 'misc'
# create a dictionary of channel names and types
chtypes_dict = dict(zip(raw.ch_names, types))

# update the channel types of our RAW data set
raw.set_channel_types(chtypes_dict)

# Apply the montage to the raw data
montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage)
# mne.viz.plot_montage(montage)

# The following plots the raw data, can be useful to spot early massive anomalies (e.g., bad channels)
# raw.plot( n_channels = 64, block = False)

###############################################################################################################
## RE REFERENCING
# Super important. Biosemi saves unreferenced data (referenced to the CMS-DRL). This means that it needs to be
# re-referenced when imported to achieve full SNR. We apply an average reference to the raw data as a PROJECTION. 
# This means that referencing is not applied directly but dealt with with a transformation matrix added to the data. 
# In plots, it will be automatically applied. 
# This is important because this transformation matrix is updated at every modification of the data 
###############################################################################################################

## Re referencing to the average
raw = raw.set_eeg_reference(ref_channels = 'average', ch_type = 'eeg', projection = True)

###############################################################################################################
## FILTERING DATA
###############################################################################################################

# Bandpass firwin filter 1 - 40 Hz
filt = raw.filter(1, 40, n_jobs = 1, fir_design = 'firwin')   # mne tutorial recommends high pass at 1 Hz

# Visualization if desired
# filt.plot_psd(fmax=30)
# Check for major bad channels 
filt.plot( n_channels = 64, block = False)

###############################################################################################################
## READING EVENTS
###############################################################################################################

# read events from the filtered dataset
events = mne.find_events(filt, stim_channel='Status', shortest_event=1)
print(events[:5])  # show the first 5

# Sometimes the value the value of the first trigger is +4. Therefore I correct it manually (but it's not very relevant because it's the encoding)
# if events[0, 1] == 4:
#     events[0, 2] = events[0, 2] - 4

# Creating the list of RETROCUES events (they end with digit 2, should be 480)
ret = (events[:, 2] == 12) | (events[:, 2] == 22) | (events[:, 2] == 32) | (events[:, 2] == 42) | (events[:, 2] == 52) | (events[:, 2] == 62) | (events[:, 2] == 72) | (events[:, 2] == 82)
retros = events[np.where(ret)]

# Dictionary with all the events specifications
event_dict = {'I_LL_Encoding': 11, 'I_LL_Retrocue': 12, 'I_LL_Probe': 13, 'I_LL_Correct': 14, 'I_LL_Wrong': 15, 'I_LL_Catch': 16, 'I_LL_Catch_Correct': 17, 'I_LL_Catch_Wrong':18,
              'I_LR_Encoding': 21, 'I_LR_Retrocue': 22, 'I_LR_Probe': 23, 'I_LR_Correct': 24, 'I_LR_Wrong': 25, 'I_LR_Catch': 26, 'I_LR_Catch_Correct': 27, 'I_LR_Catch_Wrong':28,
              'I_RL_Encoding': 31, 'I_RL_Retrocue': 32, 'I_RL_Probe': 33, 'I_RL_Correct': 34, 'I_RL_Wrong': 35, 'I_RL_Catch': 36, 'I_RL_Catch_Correct': 37, 'I_RL_Catch_Wrong':38,
              'I_RR_Encoding': 41, 'I_RR_Retrocue': 42, 'I_RR_Probe': 43, 'I_RR_Correct': 44, 'I_RR_Wrong': 45, 'I_RR_Catch': 46, 'I_RR_Catch_Correct': 47, 'I_RR_Catch_Wrong':48,
              'M_LL_Encoding': 51, 'M_LL_Retrocue': 52, 'M_LL_Probe': 53, 'M_LL_Correct': 54, 'M_LL_Wrong': 55, 'M_LL_Catch': 56, 'M_LL_Catch_Correct': 57, 'M_LL_Catch_Wrong':58,
              'M_LR_Encoding': 61, 'M_LR_Retrocue': 62, 'M_LR_Probe': 63, 'M_LR_Correct': 64, 'M_LR_Wrong': 65, 'M_LR_Catch': 66, 'M_LR_Catch_Correct': 67, 'M_LR_Catch_Wrong':68,
              'M_RL_Encoding': 71, 'M_RL_Retrocue': 72, 'M_RL_Probe': 73, 'M_RL_Correct': 74, 'M_RL_Wrong': 75, 'M_RL_Catch': 76, 'M_RL_Catch_Correct': 77, 'M_RL_Catch_Wrong':78,
              'M_RR_Encoding': 81, 'M_RR_Retrocue': 82, 'M_RR_Probe': 83, 'M_RR_Correct': 84, 'M_RR_Wrong': 85, 'M_RR_Catch': 86, 'M_RR_Catch_Correct': 87, 'M_RR_Catch_Wrong':88}

###############################################################################################################
## EPOCHING
###############################################################################################################

# Loading behavioral psychopy data to add as metadata
for file in os.listdir(data_path):
    if (file.endswith(".csv")) and (file.startswith("BindEEG_")):
        metadata = pd.read_csv(file)

# Super important: Normally the average reference projection would be automatically applied here. By adding
# proj = False we postpone the application of the projection, so that we can apply it later, after dropping
# potentially bad channels and after ICA. 
    
# Epoching (demeaning to the average of the whole epoch included)
epochs = mne.Epochs(filt, retros, event_id = {'I_LL':12, 'I_LR': 22, 'I_RL': 32, 'I_RR': 42, 'M_LL':52, 'M_LR': 62, 'M_RL': 72, 'M_RR': 82}, 
                    proj = False, baseline = (None, None), tmin = -1, tmax = 2.5, metadata = metadata, preload = True)

# Downsampling to 512 Hz (I can always downsample later - mne is fast anyway)
epochs.decimate(2)

# This plots all channels of each epoch, for visual rejection
# Bottom right of the figure, you can activate/deactivate the average projection
epochs.plot(n_epochs = 5, picks = 'eeg', n_channels = 64, block = False)

# plotting summary of all epochs
# epochs.plot_image(picks = 'eeg', combine = 'mean')

# this plots all epochs for a specific channel, can be useful if you suspect a bad channel
# epochs.plot_image('Fz', cmap='interactive')


###############################################################################################################
## BAD CHANNEL INTERPOLATION (if need be)
###############################################################################################################

epochs = epochs.interpolate_bads()
# now epochs.info['bads'] should be empty

###############################################################################################################
## APPLY REFERENCE PROJECTOR
# Now that the data are cleaned and bad channels are interpolated
# It is recommended (by Cohen and some threads on EEGlab mailing list) 
# to rereference before ICA
###############################################################################################################

epochs.apply_proj()
epochs.plot_image(picks = 'eeg', combine = 'mean')

###############################################################################################################
## ICA
###############################################################################################################

# create ICA object with desired parameters (99% of variance)
ica = mne.preprocessing.ICA(n_components = 0.999999, max_iter = 500)
# do ICA decomposition
ica.fit(epochs, decim = 2)

# plotting all the components
ica.plot_components()

# plot the properties of a single component (e.g. to check its frequency profile)
ica.plot_properties(epochs, picks=[0, 3], psd_args={'fmax': 35.}), 

# components to remove
ica.exclude = [0, 3]

# apply ICA to data
ica.apply(epochs)

###############################################################################################################
## Visual checks of data quality
###############################################################################################################

epochs.plot_psd(fmax=40., tmin = 0, tmax = 1.8, average=False, spatial_colors=True)
epochs.plot_image(picks = 'eeg', combine = 'gfp')
epochs.plot_image(picks = 'eeg', combine = 'mean')
epochs.copy().average().apply_baseline((-0.2, 0)).plot_joint()



###############################################################################################################
## Saving clean epochs
###############################################################################################################

os.chdir(data_path)
epochs.save('Subj'+str(subj)+'-epo.fif')

## THE END!




