# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:57:54 2022

@author: alex.legariamacal
"""

import numpy as np
import pandas as pd
import copy
from scipy.ndimage import gaussian_filter1d


def spiking_peh(spike_ts, ref_ts, min_max, bin_width, return_trials = False):
    """
    Finds the spiking rate around a series of events (peri-event histogram). Function assumes same units in spiking timestamps and event timestamps.
    If return_trials, histograms of spiking rate around each event will be returned as a 2d-array in addition to the average_histogram.
    
    Parameters
    -----------
    spike_ts : array-like
        Spike times of neuron
    ref_ts : array-like
        Reference (event) timestamps
    min_max : tuple
        Time window around ref_ts in which spike_ts will be analyzed. It should be a tuple containing the left (min) and right (max) bounds. E.g. (-4,4)
    bin_width : int or float
        Bin width of the returned histogram
    reutrn_trials: bool
        If True, it returns a 2d-array with the spiking activity histogram of each event in ref_ts (rows = trials, columns = time bins)
        
    Returns
    -------
    avg_trial : 1d-array
        Histogram of average spiking activity around ref_ts, with bin widths = bin_width
    trials_hist : 2d-array
        2d-array with spiking activity around each event in ref_ts, with bin width = bin_width. Rows = trials, columns = time bin. It is returned only if
        return_trials = True
    
    """
    
    if not isinstance(spike_ts, (list, np.ndarray, pd.Series)):
        raise TypeError("Expected array-like in spike_ts parameter but got " + str(type(spike_ts)) + " instead")
        
    if not isinstance(ref_ts, (list, np.ndarray, pd.Series)):
        raise TypeError("Expected array-like in ref_ts parameter but got " + str(type(ref_ts)) + " instead")
        
    if not isinstance(min_max, tuple):
        raise TypeError("Expected tuple in min_max parameter but got " + str(type(min_max)) + " instead")
        
    if not isinstance(bin_width, (int, float)):
        raise TypeError("Expected float or int in bin_width parameter but got " + str(type(bin_width)) + " instead")
    
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
    
    if return_trials:
        trials_hist = np.array([np.histogram(trial, bins)[0] for trial in trials])
        return trials_hist, avg_trial
    else:
        return avg_trial


#Downsample function for photometry (or any other continuous variable) data
def downsample_1d(var_ts, var_vals, rate):
    """
    Downsamples a time series using np.interp
    
    Parameters
    -----------
    var_ts : array-like
        Time series timestamps
    var_vals : array-like
        Time series values
    rate : int
        Sampling rate at which the variable will be downsampled
        
    Returns
    -------
    ds_var : tuple
        Contains two 1d-arrays. The first one has the downsampled variable timestamps. Second array contains downsample values.   
    """

    if not isinstance(var_ts, (list, np.ndarray, pd.Series)):
        raise TypeError("Expected array-like in var_ts parameter but got " + str(type(var_ts)) + " instead")
    
    if not isinstance(var_vals, (list, np.ndarray, pd.Series)):
        raise TypeError("Expected array-like in var_vals parameter but got " + str(type(var_vals)) + " instead")
    
    if not isinstance(rate, (int)):
        raise TypeError("Expected int in rate parameter but got " + str(type(rate)) + " instead")
   
    ds_ts = np.linspace(var_ts.min(), var_ts.max(), (var_ts.max()-var_ts.min())*rate)
    ds_vals = np.interp(ds_ts, var_ts, var_vals)
    
    ds_var = (ds_ts, ds_vals)
    
    return ds_var


#Peri-event histogram for continuous values.
def contvar_peh(var_ts, var_vals, ref_ts, min_max, bin_width = False, return_trials = True):
    """
    Finds the time series value around a series of events (peri-event histogram). Function assumes same units time series timestamps and reference timestamps.
    If return_trials, histograms of Time series around each event will be returned as a 2d-array in addition to the average histogram.
    
    Parameters
    -----------
    var_ts : array-like
        Timestamps of the time series
    var_vals : array-like
        Values of the time series
    ref_ts : array-like
        Reference (event) timestamps
    min_max : tuple
        Time window around ref_ts in which spike_ts will be analyzed. It should be a tuple containing the left (min) and right (max) bounds. E.g. (-4,4)
    bin_width : int or float
        Bin width of the returned histogram
    reutrn_trials: bool
        If True, it returns a 2d-array with the times series values around each event in ref_ts (rows = trials, columns = time bins)
        
    Returns
    -------
    avg_trial : 1d-array
        Histogram of average spiking activity around ref_ts, with bin widths = bin_width
    trials_hist : 2d-array
        2d-array with spiking activity around each event in ref_ts, with bin width = bin_width. Rows = trials, columns = time bin. It is returned only if
        return_trials = True
    
    """
    if not isinstance(var_ts, (list, np.ndarray, pd.Series)):
        raise TypeError("Expected array-like in var_ts parameter but got " + str(type(var_ts)) + " instead")
        
    if not isinstance(var_vals, (list, np.ndarray, pd.Series)):
        raise TypeError("Expected array-like in var_vals parameter but got " + str(type(var_vals)) + " instead")
        
    if not isinstance(ref_ts, (list, np.ndarray, pd.Series)):
        raise TypeError("Expected array-like in ref_ts parameter but got " + str(type(ref_ts)) + " instead")
        
    if not isinstance(min_max, tuple):
        raise TypeError("Expected tuple in min_max parameter but got " + str(type(min_max)) + " instead")
        
    if not isinstance(bin_width, (int, float)):
        raise TypeError("Expected float or int in bin_width parameter but got " + str(type(bin_width)) + " instead")
    
    if bin_width:
        rate = bin_width
        ds_ts, ds_vals = downsample_1d(var_ts, var_vals, int(1/bin_width))
    
    else:
        rate = np.diff(var_ts).mean()
        ds_ts, ds_vals = (np.array(var_ts), np.array(var_vals))       
        
    left_idx = int(min_max[0]/rate)
    right_idx = int(min_max[1]/rate)
    
    all_idx = np.searchsorted(ds_ts,ref_ts, "right")   
    all_trials = np.vstack([ds_vals[idx+left_idx:idx+right_idx] for idx in all_idx])
    avg_trial = all_trials.mean(axis=0)
    
    if return_trials == True:
        return all_trials, avg_trial
    else:
        return avg_trial
    

def zscore_df(data):
    """
    Z-scores each column of a pandas dataframe or numpy 2d-array
    
    Parameters
    -----------
    data : np.2darray or pd.DataFrame
        Data where the columns will be z-scored.
        
    Returns
    -------
    data_norm: pd.DataFrame
        Normalized (z-scored) data.
    """
    
    data_norm = copy.deepcopy(data)
    
    if isinstance(data, np.ndarray):
        for i in range(data.shape[1]):
            norm_column = (data_norm[:,i] - data_norm[:,i].mean()) / data_norm[:,i].std()
            data_norm[:,i] = norm_column
    
    elif isinstance(data, pd.DataFrame):
        for column in data_norm:
            data_norm[column] = (data_norm[column] - np.mean(data_norm[column])) / data_norm[column].std()
        
    else:
        raise TypeError("Expected array-like in bin_width parameter but got " + str(type(data)) + " instead")
    
    return data_norm

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

#Smooth every column of a dataframe using a gaussian filter
def smooth_units(data, sigma):
    for unit in data:
        data[unit] = gaussian_filter1d(data[unit],sigma)
        
    return data

