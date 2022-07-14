
import numpy as np
import sys
sys.path.append('C:\\Users\\alex.legariamacal\\Box\\Kravitz Lab Box Drive\\Alex\\communal_code')
import nexfile

reader = nexfile.Reader(useNumpy=True)

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