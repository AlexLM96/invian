# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:57:54 2022

@author: alex.legariamacal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import copy
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import linregress
#from caiman.source_extraction.cnmf import deconvolution

import sys
sys.path.append('C:\\Users\\alex.legariamacal\\Box\\Kravitz Lab Box Drive\\Alex\\communal_code')
import nexfile
import kl_classes2 as kl

reader = nexfile.Reader(useNumpy=True)

#%%

#Peri-event histogram analysis for neurons
def spiking_peh(spike_ts, ref_ts, min_max, bin_width, return_trials = False, raw_trials = False):
    trials = []
    all_trials = []
    for event in ref_ts:
        c_interval = (event+min_max[0], event+min_max[1])
        #print(c_interval)
        c_spikes = spike_ts[np.logical_and(spike_ts >= c_interval[0], spike_ts <= c_interval[1])]
        #print(c_spikes.shape)
        #d
        trials.append(c_spikes - event)
        all_trials += list(c_spikes - event)
        
    bins = np.linspace(min_max[0], min_max[1], int((min_max[1]-min_max[0])/bin_width))
    
    avg_trial = np.histogram(all_trials, bins)[0] / len(ref_ts)
    #print(avg_trial)
    
    if return_trials:
        trials_hist = np.array([np.histogram(trial, bins)[0] for trial in trials])
        if return_trials and raw_trials:
            return trials_hist, trials, avg_trial
        else:
            return trials_hist, avg_trial
    
    elif raw_trials and not return_trials:
        return trials, avg_trial
    
    else:
        return avg_trial
    
#%%

#Downsample function for photometry (or any other continuous variable) data
def downsample_1d(var_ts, var_vals,bin_width):
    ts = pd.to_timedelta(var_ts, unit = 's')
    bin_ts = str(bin_width) + "S"
    var_series = pd.Series(var_vals)
    var_series.index = ts
    ds_var = var_series.resample(bin_ts).mean()
    
    return np.array(ds_var.index.total_seconds()), np.array(ds_var)

#%%

#Peri-event histogram for continuous values.
def contvar_peh(var_ts, var_vals, ref_ts, min_max, bin_width = False):
    
    if bin_width:
        ds_ts = np.linspace(var_ts.min(), var_ts.max(), int((var_ts.max()-var_ts.min())/bin_width))
        ds_vals = np.interp(ds_ts, var_ts, var_vals)
        rate = bin_width
    
    else:
        rate = np.diff(var_ts).mean()
        ds_ts, ds_vals = (np.array(var_ts), np.array(var_vals))       
        
    left_idx = int(min_max[0]/rate)
    right_idx = int(min_max[1]/rate)
    
    all_idx = np.searchsorted(ds_ts,ref_ts, "right")   
    all_trials = np.vstack([ds_vals[idx+left_idx:idx+right_idx] for idx in all_idx])
    
    return all_trials

#%%

#Zscore every column of a dataframe
def zscore_df(df):
    for column in df:
        df[column] = (df[column] - np.mean(df[column])) / np.std(df[column])
        
    return df

#%%

#Z-scoring every column of a dataframe to a specific baseline
def zscore_tobaseline(df, baseline = (0,40)):
    start = baseline[0]
    end = baseline[1]
    for column in df:
        baseline = df[column].iloc[start:end]
        baseline_avg = np.mean(baseline)
        baseline_std = np.std(baseline)
        if baseline_std == 0:
            None
        else:
            column_zscore = (df[column] - baseline_avg) / baseline_std
            df[column] = column_zscore
        
    return df

#%%

#Smooth every column of a dataframe using a gaussian filter
def smooth_units(data, sigma):
    for unit in data:
        data[unit] = gaussian_filter1d(data[unit],sigma)
        
    return data

#%%

#Subtracting a baseline from every column of a dataframe
def subtract_baseline(df, baseline = (0,40)):
    start = baseline[0]
    end = baseline[1]
    for column in df:
        baseline = df[column].iloc[start:end]
        baseline_avg = np.mean(baseline)
        df[column] = df[column] - baseline_avg
        
    return df

#%%

#Get neuron timestamps from nex5 file
def get_neurons(file, neuron_names = "all", read_file = True):
    if read_file == True:
        c_data = reader.ReadNexFile(file)
    else:
        c_data = file
    #c_header = c_data["FileHeader"]
    c_vars = c_data["Variables"]
    neuron_data = {}
    if neuron_names == "all":
        for var in c_vars:
            if var["Header"]["Type"] == 0:
                neuron_ts = var["Timestamps"]
                neuron_data[var["Header"]["Name"]] = np.array(neuron_ts)
    else:
        for neuron in neuron_names:
            neuron_ts = [doc_var["Timestamps"] for doc_var in c_vars if doc_var["Header"]["Name"] == neuron][0]
            neuron_data[neuron] = neuron_ts             
    
    return neuron_data

#%%

#Get event timestamps from nex5 file
def get_events(file, event_names = "all", read_file = True):
    if read_file == True:
        c_data = reader.ReadNexFile(file)
    else:
        c_data = file
    #c_header = c_data["FileHeader"]
    c_vars = c_data["Variables"]
    event_data = {}
    if event_names == "all":
        for var in c_vars:
            if var["Header"]["Type"] == 1:
                var_name = var["Header"]["Name"]
                event_ts = var["Timestamps"]
                event_data[var_name] = event_ts
    else:
        for event in event_names:
            event_ts = [doc_var["Timestamps"] for doc_var in c_vars if doc_var["Header"]["Name"] == event][0]
            event_data[event] = event_ts             
    
    return event_data

#%%

#Get timestamps and values for continuous values from nex5 file
def get_contvars(file, contvar_names = "all", read_file = True):
    if read_file == True:
        c_data = reader.ReadNexFile(file)
    else:
        c_data = file
    c_header = c_data["FileHeader"]
    c_vars = c_data["Variables"]
    c_end = c_header["End"]
    
    contvar_data = {}
    if contvar_names == "all":
        for var in c_vars:
            if var["Header"]["Type"] == 5:
                contvar_vals = var["ContinuousValues"]
                contvar_ts = np.linspace(0,c_end, len(contvar_vals))
                contvar_data[var] = (contvar_ts,contvar_vals)
    else:
        for contvar in contvar_names:
            contvar_vals = [doc_var["ContinuousValues"] for doc_var in c_vars if doc_var["Header"]["Name"] == contvar][0]
            contvar_ts = np.linspace(0,c_end, len(contvar_vals))
            contvar_data[contvar] = (contvar_ts, contvar_vals)
        
    return contvar_data

