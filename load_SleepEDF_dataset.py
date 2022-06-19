


Directory = "E:\\"

data = pyedflib.EdfReader(Directory+"sc4002e0.rec")
eeg1 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"sc4002e0.hyp")
h1 = data.readSignal(0)
h1[h1 == 4] = 3
del data
eeg1, h1 = trimming(eeg1, h1)
eeg1=cut_amp(eeg1,150)
eeg1 = butter_bandpass_filter(eeg1, 0.1, 49, fs = 100, order=3)



data = pyedflib.EdfReader(Directory+"sc4012e0.rec")
eeg2 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"sc4012e0.hyp")
h2 = data.readSignal(0)
h2[h2 == 4] = 3
del data
eeg2, h2 = trimming(eeg2, h2)
eeg2=cut_amp(eeg2,150)
eeg2 = butter_bandpass_filter(eeg2, 0.1, 49, fs = 100, order=3)


data = pyedflib.EdfReader(Directory+"sc4102e0.rec")
eeg3 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"sc4102e0.hyp")
h3 = data.readSignal(0)
h3[h3 == 4] = 3
del data
eeg3, h3 = trimming(eeg3, h3)
eeg3=cut_amp(eeg3,150)
eeg3 = butter_bandpass_filter(eeg3, 0.1, 49, fs = 100, order=3)





data = pyedflib.EdfReader(Directory+"sc4112e0.rec")
eeg4 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"sc4112e0.hyp")
h4 = data.readSignal(0)
h4[h4 == 4] = 3
del data
eeg4, h4 = trimming(eeg4, h4)
eeg4=cut_amp(eeg4,150)
eeg4 = butter_bandpass_filter(eeg4, 0.1, 49, fs = 100, order=3)



data = pyedflib.EdfReader(Directory+"st7022j0.rec")
eeg5 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"st7022j0.hyp")
h5 = data.readSignal(0)
h5[h5 == 4] = 3
del data
eeg5, h5 = trimming(eeg5, h5)
eeg5=cut_amp(eeg5,150)
eeg5 = butter_bandpass_filter(eeg5, 0.1, 49, fs = 100, order=3)


data = pyedflib.EdfReader(Directory+"st7052j0.rec")
eeg6 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"st7052j0.hyp")
h6 = data.readSignal(0)
h6[h6 == 4] = 3
del data
eeg6, h6 = trimming(eeg6, h6)
eeg6=cut_amp(eeg6,150)
eeg6 = butter_bandpass_filter(eeg6, 0.1, 49, fs = 100, order=3)



data = pyedflib.EdfReader(Directory+"st7121j0.rec")
eeg7 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"st7121j0.hyp")
h7 = data.readSignal(0)
h7[h7 == 4] = 3
del data
eeg7, h7 = trimming(eeg7, h7)
eeg7=cut_amp(eeg7,150)
eeg7 = butter_bandpass_filter(eeg7, 0.1, 49, fs = 100, order=3)


data = pyedflib.EdfReader(Directory+"st7132j0.rec")
eeg8 = data.readSignal(0)
del data
data = pyedflib.EdfReader(Directory+"st7132j0.hyp")
h8 = data.readSignal(0)
h8[h8 == 4] = 3
del data
eeg8, h8 = trimming(eeg8, h8)
eeg8=cut_amp(eeg8,150)
eeg8 = butter_bandpass_filter(eeg8, 0.1, 49, fs = 100, order=3)



#centering 
eeg1 = eeg1 - np.mean(eeg1)
eeg2 = eeg2 - np.mean(eeg2)
eeg3 = eeg3 - np.mean(eeg3)
eeg4 = eeg4 - np.mean(eeg4)
eeg5 = eeg5 - np.mean(eeg5)
eeg6 = eeg6 - np.mean(eeg6)
eeg7 = eeg7 - np.mean(eeg7)
eeg8 = eeg8 - np.mean(eeg8)

eeg1 /= np.std(eeg1)
eeg2 /= np.std(eeg2)
eeg3 /= np.std(eeg3)
eeg4 /= np.std(eeg4)
eeg5 /= np.std(eeg5)
eeg6 /= np.std(eeg6)
eeg7 /= np.std(eeg7)
eeg8 /= np.std(eeg8)


EEG = np.concatenate((eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, eeg8), axis=0)
H = np.concatenate((h1, h2, h3, h4, h5, h6, h7, h8), axis=0)



n = 0
EEG1 = []
while (n < len(H)):
    
    EEG1.append(EEG[n*3000:(n+1)*3000])
    n += 1
  
EEG1 = np.array(EEG1)


del eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, eeg8










