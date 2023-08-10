# import librairy
import numpy as np
import os
import pandas as pd
from scipy.signal import welch
import scipy.io
import matplotlib.pyplot as plt
import plot
import mne
from eeglabio.utils import export_mne_raw
import asrpy
from plot import show_figure


#passband
Fpass = [0.5,40]

#Sampling Frequency
Fs = 256

#Nyquist Frequency
Fn = Fs/2
resolution=0.5
noverlap=0
nfft=round(Fs/resolution)
window=np.hanning(nfft)
L = 1200

#variable that describe the differents path to save or uploads files (Only these variables must be change when we use a different computer)
directory = r'C:\Users\estelle\Documents\donnée matlab'
path =  r'C:\Users\estelle\Documents\code python'

#path where is the csv
data_files = [f for f in os.listdir(directory) if f.endswith('-data.csv')] #find the name of data files
event_files = [f for f in os.listdir(directory) if f.endswith('-events.csv')] #find the name of event files

#create folder where store raw data matrix
if not os.path.isdir(path+'\\raw_data'):
    os.makedirs(path+'\\raw_data')

#folder where store data after using bandpass
if not os.path.isdir(path+'\\bandpass'):
    os.makedirs(path+'\\bandpass')

#folder where store data after using asr
if not os.path.isdir(path+'\\asr'):
    os.makedirs(path+'\\asr')



for i in range(len(data_files)):
    
    #find the name of the files
    kd = data_files[i].find('.')
    name_data = data_files[i][:kd]

    ke = event_files[i].find('.')
    name_events = event_files[i][:ke]
    channels_names = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPZ", "CP4"] 
    #open the data
    bitbrain = pd.read_csv(directory+'\\'+data_files[i])

    #extract the times vector
    bitbrain_data_t = bitbrain['Time'] #time vector
    bitbrain_data_t_corrected = bitbrain_data_t-bitbrain_data_t[0] #time vector corrected so it's begin at 0s
    bitbrain_data = bitbrain

    #save the data
    bitbrain_data.to_csv(path+'\\raw_data\\'+name_data+'_raw.csv')

    #we make the same but with the event file
    csv_event = pd.read_csv(directory+'\\'+event_files[i],header=None)
    events = csv_event.iloc[:,0]
    events_corrected = events - bitbrain_data_t[0]
    

    ################################ apply bandpass fonction
    
    bb_data = pd.DataFrame(columns=channels_names)
    for col in range(1,len(channels_names)+1):
        a = mne.filter.filter_data(bitbrain_data.iloc[:,col].to_numpy(),method='fir',fir_window='hann',filter_length=128,sfreq=Fs,l_freq=None,h_freq=35)
        bb_data[channels_names[col-1]]=a

    bb_data.insert(0,'Time',value=bitbrain_data_t)
 
    #find the different position in dataframe

    #position of the first oe(open eyes) event
    oe1 = abs(bitbrain_data_t_corrected-events_corrected[3])
    min_difference_oe1,position_oe1 = min(oe1),np.argmin(oe1)

    #position of the ce(close eyes) event
    ce = abs(bitbrain_data_t_corrected - events_corrected[4])
    p1 = abs(bitbrain_data_t_corrected - events_corrected[5])#first pause
    min_difference_ce,position_ce = min(ce),np.argmin(ce)
    min_difference_p1,position_p1 = min(p1),np.argmin(p1)

    #position where start the first task event
    t1 = abs(bitbrain_data_t_corrected - events_corrected[6])
    p2 = abs(bitbrain_data_t_corrected - events_corrected[126]) #second pause
    min_difference_t1,position_t1 = min(t1),np.argmin(t1)
    min_difference_p2,position_p2 = min(p2),np.argmin(p2)

    #position where start the second task event
    t2 = abs(bitbrain_data_t_corrected - events_corrected[127])
    p3 = abs(bitbrain_data_t_corrected - events_corrected[246]) #third pause
    min_difference_t2,position_t2 = min(t2), np.argmin(t2)
    min_difference_p3,position_p3 = min(p3),np.argmin(p3)

    #position of the second oe event
    oe3 = abs(bitbrain_data_t_corrected - events_corrected[247])
    end = abs(bitbrain_data_t_corrected - events_corrected[248])
    min_difference_oe3, position_oe3 = min(oe3), np.argmin(oe3)
    min_difference_end, position_end = min(end), np.argmin(end)

    #calculate extra position to divide the events 
    position_oe2 = (position_oe1 + position_ce)//2
    position_oe4 = (position_oe3+position_end)//2
    position_ce2 = (position_ce + position_p1)//2

    samples_60_seconds = 60*Fs #number of samples in 60seconds

    #create folders to store the differents figures
    if not os.path.isdir(path+'\\'+name_data+'_figures'):
        os.makedirs(path+'\\'+name_data+'_figures')
        os.makedirs(path+'\\'+name_data+'_figures\\figures_asr')
        os.makedirs(path+'\\'+name_data+'_figures\\figures_sinasr')

    #PSD figures (before ASR)

    #10E
    plot.show_figure("fig1",position_oe1,position_oe2,path,name_data,'10E',bb_data,Fs,nfft)
    plt.close()

    #20E
    plot.show_figure('fig2',position_oe2,position_ce,path,name_data,"20E",bb_data,Fs,nfft)
    plt.close()

    #1CE
    plot.show_figure("fig3",position_ce,position_ce2,path,name_data,"1CE",bb_data,Fs,nfft)
    plt.close()

    #2CE
    plot.show_figure("fig4",position_ce2,position_p1,path,name_data,"2CE",bb_data,Fs,nfft)
    plt.close()
    
    #30E
    plot.show_figure("fig5",position_oe3,position_oe4,path,name_data,"30E",bb_data,Fs,nfft)
    plt.close()

    #40E
    plot.show_figure("fig6",position_oe4,position_oe4+samples_60_seconds,path,name_data,"40E",bb_data,Fs,nfft)
    plt.close()


    #########################################Events table################################################################
    type = csv_event.iloc[2:,1]
    events_eeglab = events_corrected[2:]
    latency = np.zeros((len(type),1))
    left_arrow_pos = []
    right_arrow_pos = []
    duration = np.zeros((len(type),1))

    sample_events = {"OE":120,"CE":120,"IMGTASK_START":5,"IMGTASK_RA":3,"IMGTASK_LA":3,"IMGTASK_AT":2,"PAUSE":1,"END":1}
    for h in range(len(type)):
        if type.iloc[h] in sample_events:
            d = abs(bitbrain_data_t_corrected-events_eeglab.iloc[h])
            min_difference,position = min(d),np.argmin(d)
            duration[h] = sample_events[type.iloc[h]]*Fs
            latency[h] = position

            if type.iloc[h]=="IMGTASK_RA":
                pos_right = position+1
                right_arrow_pos.append(pos_right)
            elif type.iloc[h]=="IMGTASK_LA":
                pos_left = position+1
                left_arrow_pos.append(pos_left)

    event = pd.DataFrame(columns=["latency","duration","type"])
    event["type"]= type
    event["latency"]=latency
    event["duration"]=duration

    ######################################Apply ASR #####################################################################

    ##### STEP 1 : CREATE A RAW OBJECT
    data = np.transpose(bb_data.iloc[:,1:])

    info = mne.create_info(channels_names,Fs,"eeg")
    raw = mne.io.RawArray(data, info)

    montage = mne.channels.read_custom_montage(directory+'\\Painapp.sfp')
    raw.set_montage(montage)


    ##### STEP 2 : save in .set files
    raw.save(path+'\\bandpass\\' + name_data + '.fif', overwrite=True)
    bandpass = mne.io.read_raw(path+'\\bandpass\\' + name_data + '.fif')
    export_mne_raw(bandpass,path+"\\bandpass\\"+ name_data +'.set')

    ##### STEP 3 : apply ASR 
    try : 
        raw.load_data()
        raw.set_eeg_reference()
        asr = asrpy.ASR(sfreq=raw.info["sfreq"])
        asr.fit(raw)
        
        raw = asr.transform(raw,lookahead=len(window)/2,stepsize=Fs//3)

        asr = raw.to_data_frame()
        asr.to_csv(path+'\\asr.csv')
    except IndexError:
        print("ASR doesn't work")
        
    
    