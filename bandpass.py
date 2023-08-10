import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

def bandpass(data):
    Fpass = [0.5,40]

    #Sampling Frequency
    Fs = 256

    time = data['Time']
    channels_names = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPZ", "CP4"]
    data_bandpass = pd.DataFrame(columns=channels_names)
    for col in range(1,len(channels_names)+1):
        a = mne.filter.filter_data(data.iloc[:,col].to_numpy(),method='fir',fir_window='hann',filter_length=128,sfreq=Fs,l_freq=None,h_freq=Fpass[1])
        data_bandpass[channels_names[col-1]]=a

    data_bandpass.insert(0,'Time',value=time)
    
    return data_bandpass