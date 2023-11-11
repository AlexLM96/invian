from .nexfile import *
import numpy as np

reader = Reader(useNumpy=True)

#%%

class Header():

    def __init__(self, fileloc):
        nexfile_header = reader.ReadNex5FileHeader(fileloc)
        self._metadata = nexfile_header["FileHeader"]
        self._variables = nexfile_header["Variables"]

    @property
    def variables(self):
        var_types = {
        0: 'neuron',
        1: 'event',
        2: 'interval',
        3: 'waveform',
        4: 'population_vector',
        5: 'continuous',
        6: 'marker'
        }
        vars = {var_types[v_type]: [] for v_type in var_types.keys()}
        for var in self._variables:
            c_var_type = var["Header"]["Type"]
            vars[var_types[c_var_type]].append(var["Header"]["Name"])

        return vars
    
    @property
    def metadata(self):
        return self._metadata




#%%
#Read header
def read_header(fileloc):
    return Header(fileloc)

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
"""
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
"""

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
                
                c_sr = 1/var["Header"]["SamplingRate"] 
                frag_idx = var["FragmentIndexes"]
                frag_ts = var["FragmentTimestamps"]
                frag_count = var["FragmentCounts"]
                c_ts = np.zeros(len(contvar_vals))
                c_ts[frag_idx] = frag_ts
                
                for i in range(len(frag_idx)):
                    c_frag_ts = frag_ts[i]
                    c_frag_idx = frag_idx[i]
                    c_frag_count = frag_count[i]
                    
                    c_gap = np.arange(c_frag_count-1)+1
                    c_gap_f = (c_gap*c_sr) + c_frag_ts
                    c_gap_idx = c_gap+c_frag_idx
                    
                    c_ts[c_gap_idx] = c_gap_f
                        
                contvar_ts = c_ts
                contvar_data[var] = (contvar_ts,contvar_vals)
    else:
        for contvar in contvar_names:
            c_var = [doc_var for doc_var in c_vars if doc_var["Header"]["Name"] == contvar][0]
            
            contvar_vals = c_var["ContinuousValues"]
            
            c_sr = 1/c_var["Header"]["SamplingRate"]
            frag_idx = c_var["FragmentIndexes"]
            frag_ts = c_var["FragmentTimestamps"]
            frag_count = c_var["FragmentCounts"]
            c_ts = np.zeros(len(contvar_vals))
            c_ts[frag_idx] = frag_ts

            for i in range(len(frag_idx)):
                c_frag_ts = frag_ts[i]
                c_frag_idx = frag_idx[i]
                c_frag_count = frag_count[i]
                
                c_gap = np.arange(c_frag_count-1)+1
                c_gap_f = (c_gap*c_sr) + c_frag_ts
                c_gap_idx = c_gap+c_frag_idx
                
                c_ts[c_gap_idx] = c_gap_f
                    
            contvar_ts = c_ts
            contvar_data[contvar] = (contvar_ts,contvar_vals)

        
    return contvar_data