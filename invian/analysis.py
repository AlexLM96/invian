
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


#%%

#Peri-event histogram analysis for neurons
def spiking_peh(spike_ts, ref_ts, min_max, bin_width, return_trials = False):
    r"""
    Function to perform a peri-event histogram of spiking activity.
    
    Parameters
    ----------
    spike_ts : array-like
        Spike timestamps
    ref_ts : array-like
        Reference events that spiking will be aligned to
    min_max : tuple
        Time window in seconds around ref_ts to be analyzed
    bin_width : float
        Bin width in seconds
    return_trials : bool
        If true, returns a 2d-array where each row is a trial and each column is a time bin

    Returns
    ---------
    trials_hist : 2d-array
        Spiking activity around each timestamp in ref_ts
    avg_trial : 1d-array
        Average spiking activity around ref_ts
    """
    
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
        return trials_hist
    
    else:
        return avg_trial
    
#%%

#Downsample function for photometry (or any other continuous variable) data
def downsample_1d(var_ts, var_vals,bin_width):
    r"""
    Downsamples 1d time series
    
    Parameters
    ----------
    var_ts : array-like
        Variable timestamps
    var_vals : array-like
        Variable timestamps continuous values
    bin_width : float
        Bin width in seconds
    return_trials : bool
        If true, returns a 2d-array where each row is a trial and each column is a time bin

    Returns
    ---------
    trials_hist : 2d-array
        Spiking activity around each timestamp in ref_ts
    avg_trial : 1d-array
        Average spiking activity around ref_ts
    """
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
    print(start, end)
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


"""
def plot_peh(all_trials, xrange, title = None):
    fig, ax = plt.subplots(2,1)
    sns.heatmap(all_trials, cbar = False, xticklabels = False, yticklabels = False, ax = ax[0])
    ax[1].plot(np.linspace(xrange[0], xrange[1], all_trials.shape[1]), all_trials.mean(axis = 0))
    ax[1].set_xlim(xrange[0], xrange[1])
    
    if title != None:
        ax[0].set_title(title)
    
    return fig, ax
"""