import pandas as pd
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
import os

### type = 0 : raw data
### typ = 1 : bandpass



def show_figure(number,start,end,path,nameData,namefigure,data,Fs,nfft,type):
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
    plt.axis([0,60,0,15])
    plt.xlabel('Frequency')


    plt.subplot(132)
    plt.plot(PSD.iloc[4:80,1],label = "FCz")
    plt.plot(PSD.iloc[4:80,4],label='Cz')
    plt.plot(PSD.iloc[4:80,7],label='CPz')
    plt.title("PSD central channels")
    plt.axis([0,60,0,15])
    plt.legend()
    plt.xlabel('Frequency')
    
    plt.subplot(133)
    plt.plot(PSD.iloc[4:80,2],label = "FC4")
    plt.plot(PSD.iloc[4:80,5],label='C4')
    plt.plot(PSD.iloc[4:80,8],label='CP4')
    plt.title("PSD right channels")
    plt.axis([0,60,0,15])
    plt.legend()
    plt.xlabel('Frequency')

    plt.suptitle(namefigure+" SIN ASR")
    if type==0:
        number.savefig(path+'\\'+nameData+"_figures\\figures_raw_data\\"+nameData+"_PSD_"+namefigure+"_sinasr.jpg")
    else:
        number.savefig(path+'\\'+nameData+"_figures\\figures_bandpass\\"+nameData+"_PSD_"+namefigure+"_sinasr.jpg")


def psd(data,event,data_files,type):
    data_name = data_files.split('/')
    kd = data_name[-1].find('.')
    name_data = data_name[-1][:kd]
    path = '/'.join(data_name[:-1])


    if not os.path.isdir(path+'\\'+name_data+'_figures'):
        os.makedirs(path+'\\'+name_data+'_figures')
        os.makedirs(path+'\\'+name_data+'_figures\\figures_bandpass')
        os.makedirs(path+'\\'+name_data+'_figures\\figures_raw_data')

    Fs = 256
    resolution = 0.5
    nfft = round(Fs/resolution)

    time = data['Time']
    events = event.iloc[:,0]

    
    #position of the first oe(open eyes) event
    oe1 = abs(time-events[3])
    min_difference_oe1,position_oe1 = min(oe1),np.argmin(oe1)

    #position of the ce(close eyes) event
    ce = abs(time- events[4])
    p1 = abs(time - events[5])#first pause
    min_difference_ce,position_ce = min(ce),np.argmin(ce)
    min_difference_p1,position_p1 = min(p1),np.argmin(p1)
    

    #position where start the first task event
    t1 = abs(time - events[6])
    p2 = abs(time - events[126]) #second pause
    min_difference_t1,position_t1 = min(t1),np.argmin(t1)
    min_difference_p2,position_p2 = min(p2),np.argmin(p2)

    #position where start the second task event
    t2 = abs(time - events[127])
    p3 = abs(time - events[246]) #third pause
    min_difference_t2,position_t2 = min(t2), np.argmin(t2)
    min_difference_p3,position_p3 = min(p3),np.argmin(p3)

    #position of the second oe event
    oe3 = abs(time - events[247])
    end = abs(time - events[248])
    min_difference_oe3, position_oe3 = min(oe3), np.argmin(oe3)
    min_difference_end, position_end = min(end), np.argmin(end)

    #calculate extra position to divide the events 
    position_oe2 = (position_oe1 + position_ce)//2
    position_oe4 = (position_oe3+position_end)//2
    position_ce2 = (position_ce + position_p1)//2

    samples_60_seconds = 60*Fs #number of samples in 60seconds
     
    #10E
    show_figure("fig1",position_oe1,position_oe2,path,name_data,'1OE',data,Fs,nfft,type)
    plt.close()


    #20E
    show_figure('fig2',position_oe2,position_ce,path,name_data,"2OE",data,Fs,nfft,type)
    plt.close()

    #1CE
    show_figure("fig3",position_ce,position_ce2,path,name_data,"1CE",data,Fs,nfft,type)
    plt.close()

    #2CE
    show_figure("fig4",position_ce2,position_p1,path,name_data,"2CE",data,Fs,nfft,type)
    plt.close()
    
    #30E
    show_figure("fig5",position_oe3,position_oe4,path,name_data,"3OE",data,Fs,nfft,type)
    plt.close()

    #40E
    show_figure("fig6",position_oe4,position_oe4+samples_60_seconds,path,name_data,"4OE",data,Fs,nfft,type)
    plt.close()
         


    