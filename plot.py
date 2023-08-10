import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

def show_figure(number,start,end,path,nameData,namefigure,data,Fs,nfft):
    PSD = pd.DataFrame()
    for col in range(data.shape[1]-1):
        f,psd = welch(data.iloc[start:end+1,col+1],fs = Fs,window=np.hanning(nfft),nfft = nfft)
        PSD[col] = psd
    PSD = pd.DataFrame(PSD)
    
    number = plt.figure()
    plt.subplot(131)
    plt.plot(PSD.iloc[4:80,0],label='FC3')
    plt.plot(PSD.iloc[4:80,3],label='C3')
    plt.plot(PSD.iloc[4:80,6],label='CP3')
    plt.title("PSD left channels")
    plt.legend()
    plt.axis([0,55,0,15])
    plt.xlabel('Frequency')


    plt.subplot(132)
    plt.plot(PSD.iloc[4:80,1],label = "FCz")
    plt.plot(PSD.iloc[4:80,4],label='Cz')
    plt.plot(PSD.iloc[4:80,7],label='CPz')
    plt.title("PSD central channels")
    plt.axis([0,55,0,15])
    plt.legend()
    
    plt.subplot(133)
    plt.plot(PSD.iloc[4:80,2],label = "FC4")
    plt.plot(PSD.iloc[4:80,5],label='C4')
    plt.plot(PSD.iloc[4:80,8],label='CP4')
    plt.title("PSD right channels")
    plt.axis([0,55,0,15])
    plt.legend()

    plt.suptitle(nameData+namefigure+" SIN ASR")
    number.savefig(path+'\\'+nameData+"_figures\\figures_sinasr\\"+nameData+"_PSD_"+namefigure+"_sinasr.jpg")