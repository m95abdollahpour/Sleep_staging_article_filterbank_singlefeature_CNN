from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, concatenate, Input
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization, Dropout
from tensorflow.keras.layers import Flatten, Concatenate, Activation, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, freqz, sosfilt
import os
from scipy.io import loadmat
import pyedflib
from sklearn import preprocessing
import random
from scipy import stats
import sys
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pandas import read_csv
from numpy import loadtxt
import pandas as pd
from scipy import stats
import json
from keras.models import load_model
from sklearn.utils import shuffle




def edfxRead(directory):

    '''
    Read *EDF files including EEG signals and hypnograms
    
    '''

    f = open(directory)
    file = f.read()
    f.close()
    
    f = pyedflib.EdfReader(directory)
    h =  np.array(f.readAnnotations())
    del f
    labels = np.ones((len(h[2,:])))
    n = 0
    while (n < len(h[2,:])):
    
        if (h[2,n][12] == '?'):
            labels[n] = 6
        elif (h[2,n][12] == 'W'):
            labels[n] = 0
        elif (h[2,n][12] == 'e'):   
            labels[n] = 6
        elif (h[2,n][12] == 'R'):
            labels[n] = 5
        elif (h[2,n][12] == '4'):
            labels[n] = 3
        else:
            labels[n] = int(h[2,n][12])

        n =  n + 1
    
    pos = np.float32(np.array(h[0,:]))/30
    hlength = int(pos[len(pos)-1])
    n = 0
    Labels = np.ravel(30* np.ones((1, hlength)))
    
    while (n<len(pos) - 1):
        
        a = int(pos[n])
        b = int(pos[n+1])

        if (labels[n] == 0):
            Labels[a:b] = 0
        elif (labels[n] == 1):
            Labels[a:b] = 1
        elif (labels[n] == 2):
            Labels[a:b] = 2
        elif (labels[n] == 3):
            Labels[a:b] = 3
        elif (labels[n] == 4):
            Labels[a:b] = 4
        elif (labels[n] == 5):
            Labels[a:b] = 5
        elif (labels[n] == 6):
            Labels[a:b] = 6
    
        n = n + 1
        
    return Labels





def HYPNO (hypnogram):
    '''
    Reassigning the labels in hypnogram
    
    '''
    h = []
    for i in range(int(len(hypnogram)/6)):
        h.append(int(stats.mode(hypnogram[i*6:(i+1)*6])[0][0]))
    h = np.array(h)
    h[h==0] = 10
    h[h== -1] = 10
    h[h== -2] = 10
    h[h== -3] = 10
    h[h==5] = 0
    h[h==3] = 6
    h[h==1] = 3
    h[h==6] = 1
    h = np.delete(h, np.where(h==10))
    return  h





def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    defining bandpass filter
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a



def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y




def cut_amp(data, amp, direction='up'):
    
    ''' 
    Winsorizing
    
    Parameters
    -----------
    data: 1D array (input signal)
    
    amp: 1D array (maximum amplitude value to be trimmed)

    '''
    
    if (direction == 'up'):
        i=0
        while (i<len(data)):
            
            if (data[i]>amp):
                data[i]=amp
                i=i+1
            elif (data[i]<-amp):
                data[i]=-amp
                i=i+1
            else:
                i=i+1
    elif (direction == 'down'):
        i=0
        while (i<len(data)):
            
            if (data[i]<amp):
                data[i]=amp
                i=i+1
            else:
                i=i+1
    return data



    
def trimming(eeg,hypnogram):
    '''
    Removing extra EEG data to have 
    a hypnogram and an EEG with the same length and number of epochs.
    '''
    
    n = 0
    EEG1 = []
    H1 = []
    hypnogram[hypnogram == 5] = 4
    i = 0
    EEG2 = np.ones((len(eeg)))
    while (n < len(hypnogram)):
        
        if (hypnogram[n] < 5):
            EEG1.append(eeg[n*3000:(n+1)*3000])
            H1.append(hypnogram[n])
            EEG2[i*3000:(i+1)*3000] = eeg[n*3000:(n+1)*3000]
            n += 1
            i += 1
        else:
            n += 1
                
    EEG = np.array(EEG1)
    H = np.array(H1)
    EEG1 = EEG2[:i*3000]

    return EEG1, H

