
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Lasso

#%%

#Peri-event histogram analysis for neurons
def spiking_peh(spike_ts, ref_ts, min_max, bin_width):
    r"""
    Function to perform a peri-event histogram of spiking activity.
    
    Parameters
    ----------
    spike_ts : array-like
        Spike timestamps
    ref_ts : array-like
        Reference events that spiking will be aligned to
    min_max : tuple
        Time window in seconds around ref_ts to be analyzed in seconds. E.g. (-4,8)
    bin_width : float
        Bin width in seconds

    Returns
    ---------
    trials_hists : 2d-array
        Spiking activity around each timestamp in ref_ts
    """
    if not isinstance(spike_ts,np.ndarray):
        try:
            spike_ts = np.array(spike_ts)          
        except:
            raise TypeError(f"Expected spike_ts to be of type: array-like but got {type(spike_ts)} instead")
    
    if not isinstance(ref_ts,np.ndarray):
        try:
            ref_ts = np.array(ref_ts)     
        except:
            raise TypeError(f"Expected spike_ts to be of type: array-like but got {type(spike_ts)} instead")
    
    bins = np.linspace(min_max[0], min_max[1], int((min_max[1]-min_max[0])/bin_width))
    
    left_idx = np.searchsorted(spike_ts, ref_ts+min_max[0])
    right_idx = np.searchsorted(spike_ts, ref_ts+min_max[1])
    
    raw_trial_spikes = np.array([spike_ts[left_idx[i]:right_idx[i]] for i in range(left_idx.shape[0])], dtype=object)
    trial_spikes = np.subtract(raw_trial_spikes, ref_ts)
    trial_hists = np.vstack([np.histogram(trial,bins)[0] for trial in trial_spikes])
    
    return trial_hists

    
#%%

#Downsample function for photometry (or any other continuous variable) data
def downsample_1d(var_ts, var_vals,bin_width):
    r"""
    Downsamples 1d time series
    
    Parameters
    ----------
    var_ts : array-like
        Continuous variable timestamps
    var_vals : array-like
        Continuous variable values
    bin_width : float
        Bin width of new sampling rate (i.e. 1/new_sampling_rate)
    
    Returns
    ---------
    ds_ts : 1d np.array
        downsampled timestamps
    ds_vale : 1d np.array
        downsampled values
    """
    ds_ts = np.linspace(var_ts.min(), var_ts.max(), int((var_ts.max()-var_ts.min())/bin_width))
    ds_vals = np.interp(ds_ts, var_ts, var_vals)
    
    return ds_ts, ds_vals

#%%

#Peri-event histogram for continuous values.
def contvar_peh(var_ts, var_vals, ref_ts, min_max, bin_width = False):
    r"""
    Function to perform a peri-event histogram of spiking activity.
    
    Parameters
    ----------
    var_ts : array-like
        Continuous variable timestamps
    var_vals : array-like
        Continuous variable values
    ref_ts : array-like
        Reference events that spiking will be aligned to
    min_max : tuple
        Time window in seconds around ref_ts to be analyzed in seconds. E.g. (-4,8)
    bin_width : float
        Bin width of hsitogram in seconds

    Returns
    ---------
    all_trials : 2d-array
        Continuous variable values around each timestamp in ref_ts in bin_width wide bins
    """
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
def zscore_peh(data, trials="rows"):
    
    if isinstance(data, np.ndarray):
        if trials == "rows":
            z_data = (data - data.mean(axis=1)[:,np.newaxis]) / data.std(axis=1)[:,np.newaxis]
        elif trials == "columns":
            z_data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    elif isinstance(data, pd.core.frame.DataFrame):
        if trials == "rows":
            z_data = data.subtract(data.mean(axis=1),axis=0).divide(data.mean(axis=1),axis=0)
        if trials == "columns":
            z_data = data.subtract(data.mean(axis=0),axis=1).divide(data.mean(axis=0),axis=1)

    return z_data

#%%

#Z-scoring every column of a dataframe to a specific baseline
def zscore_peh_tobaseline(df, baseline = (0,40)):
    start = baseline[0]
    end = baseline[1]
    print(start, end)
    for column in df:
        baseline = df[column].iloc[start:end]
        baseline_avg = np.mean(baseline)
        baseline_std = np.std(baseline)
        if baseline_std == 0:
            pass
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

def remove_isosbestic(gcamp_signal, isosbestic_signal):
    regr = Lasso()
    regr.fit(isosbestic_signal.reshape(-1,1), gcamp_signal.reshape(-1,1))
    isos_pred = regr.predict(isosbestic_signal.reshape(-1,1))
    
    norm_gcamp_signal = gcamp_signal - isos_pred
    
    return norm_gcamp_signal



#%%

def plot_peh(all_trials, xrange, title = None):
    fig, ax = plt.subplots(2,1)
    sns.heatmap(all_trials, cbar = False, xticklabels = False, yticklabels = False, ax = ax[0])
    ax[1].plot(np.linspace(xrange[0], xrange[1], all_trials.shape[1]), all_trials.mean(axis = 0))
    ax[1].set_xlim(xrange[0], xrange[1])
    
    if title != None:
        ax[0].set_title(title)
    
    return fig, ax
    