import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


class photometry():
    """
    Class representing a photometry object. 
    
    """
    
    def __init__(self, timestamps, signal, sampling_rate = None):
        """
        
        """
        self.timestamps = np.array(timestamps)
        self.signal = np.array(signal)
        
        if sampling_rate == None:
            self.sr = self.timestamps.diff().mean()
        else:
            self.sr = sampling_rate
    
    def butter_highpass(self, low):
        b,a = butter(3, low, btype='high', fs = self.sr)
        return b, a

    def butter_lowpass(self, high):
        b,a = butter(3, high, btype = 'low', fs = self.sr)
        return b, a
    
    def hp_filter(self, low):
        b, a = self.butter_highpass(low)
        y = filtfilt(b, a, self.signal, padtype = "even")
        return photometry(self.timestamps, y, self.sr)
    
    def lp_filter(self, high):
        b,a = self.butter_lowpass(high)
        y = filtfilt(b, a, self.signal, padtype = 'even')
        return photometry(self.timestamps, y, self.sr)
    
    def zscore(self):
        avg = np.mean(self.signal)
        std = np.std(self.signal)
        z_score = (self.signal - avg) / std
        return photometry(self.timestamps, z_score, self.sr)
        
    def plot(self):
        fig, ax = plt.subplots(figsize = (6,1))
        ax.plot(self.timestamps, self.signal)
        plt.show()