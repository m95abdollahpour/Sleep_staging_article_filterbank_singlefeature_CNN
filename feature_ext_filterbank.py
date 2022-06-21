from functions import *


'''
filterbank with 50 bandpass filters with a 1H frequency band and
extracting standard deviation as a single feature
'''


X = np.ones((len(EEG1),98))

for i in range(len(EEG1)):
    amp = np.ones((49,1))
    amp2 = np.ones((49,1))
    for j in range(49):
        e = butter_bandpass_filter(EEG1[i,:1500], j+0.001, j+1, fs = 100, order=3)
        amp[j] = np.std(e)

    for j in range(49):
        e = butter_bandpass_filter(EEG1[i,1500:3000], j+0.001, j+1, fs = 100, order=3)
        amp2[j] = np.std(e)        
    
    X[i] = np.concatenate((np.ravel(amp), np.ravel(amp2)))

 








