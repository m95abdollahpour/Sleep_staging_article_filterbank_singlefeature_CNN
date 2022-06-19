import pyedflib
import numpy as np



'''
data.readSignal(0) : Fpz-Cz
data.readSignal(1) : Pz-Oz
data.readSignal(2) : EOG horizontal

f1 = pyedflib.EdfReader(directory)
eeg1 = f1.readSignal(0)
n = f1.signals_in_file
signal_labels = data.getSignalLabels()
'''

Directory = "E:\\" # Directory which contains the sleep EDFx files

data = pyedflib.EdfReader(Directory+"SC4001E0-PSG.edf")
eeg1=np.float32(data.readSignal(0))
del data
h1 = edfxRead(Directory+"SC4001EC-Hypnogram.edf")
eeg1 = eeg1[:np.uint(len(h1)*30*100)]
eeg1, h1 = trimming(eeg1, h1)
eeg1 -= np.mean(eeg1)
eeg1 /= np.std(eeg1)



data = pyedflib.EdfReader(Directory+"SC4021E0-PSG.edf")
eeg2=np.float32(data.readSignal(0))
del data
h2 = edfxRead(Directory+"SC4021EH-Hypnogram.edf")
eeg2 = eeg2[:np.uint(len(h2)*30*100)]
eeg2, h2 = trimming(eeg2, h2)
eeg2 -= np.mean(eeg2)
eeg2 /= np.std(eeg2)


data = pyedflib.EdfReader(Directory+"SC4031E0-PSG.edf")
eeg3=np.float32(data.readSignal(0))
del data
h3 = edfxRead(Directory+"SC4031EC-Hypnogram.edf")
eeg3 = eeg3[:np.uint(len(h3)*30*100)]
eeg3, h3 = trimming(eeg3, h3)
eeg3 -= np.mean(eeg3)
eeg3 /= np.std(eeg3)





data = pyedflib.EdfReader(Directory+"SC4061E0-PSG.edf")
eeg4=np.float32(data.readSignal(0))
del data
h4 = edfxRead(Directory+"SC4061EC-Hypnogram.edf")
eeg4 = eeg4[:np.uint(len(h4)*30*100)]
eeg4, h4 = trimming(eeg4, h4)
eeg4 -= np.mean(eeg4)
eeg4 /= np.std(eeg4)


data = pyedflib.EdfReader(Directory+"SC4071E0-PSG.edf")
eeg5=np.float32(data.readSignal(0))
del data
h5 = edfxRead(Directory+"SC4071EC-Hypnogram.edf")
eeg5 = eeg5[:np.uint(len(h5)*30*100)]
eeg5, h5 = trimming(eeg5, h5)
eeg5 -= np.mean(eeg5)
eeg5 /= np.std(eeg5)


data = pyedflib.EdfReader(Directory+"SC4081E0-PSG.edf")
eeg6=np.float32(data.readSignal(0))
del data
h6 = edfxRead(Directory+"SC4081EC-Hypnogram.edf")
eeg6 = eeg6[:np.uint(len(h6)*30*100)]
eeg6, h6 = trimming(eeg6, h6)
eeg6 -= np.mean(eeg6)
eeg6 /= np.std(eeg6)




data = pyedflib.EdfReader(Directory+"SC4141E0-PSG.edf")
eeg7=np.float32(data.readSignal(0))
del data
h7 = edfxRead(Directory+"SC4141EU-Hypnogram.edf")
eeg7 = eeg7[:np.uint(len(h7)*30*100)]
eeg7, h7 = trimming(eeg7, h7)
eeg7 -= np.mean(eeg7)
eeg7 /= np.std(eeg7)



data = pyedflib.EdfReader(Directory+"SC4151E0-PSG.edf")
eeg8=np.float32(data.readSignal(0))
del data
h8 = edfxRead(Directory+"SC4151EC-Hypnogram.edf")
eeg8 = eeg8[:np.uint(len(h8)*30*100)]
eeg8, h8 = trimming(eeg8, h8)
eeg8 -= np.mean(eeg8)
eeg8 /= np.std(eeg8)





data = pyedflib.EdfReader(Directory+"SC4181E0-PSG.edf")
eeg9=np.float32(data.readSignal(0))
del data
h9 = edfxRead(Directory+"SC4181EC-Hypnogram.edf")
eeg9 = eeg9[:np.uint(len(h9)*30*100)]
eeg9, h9 = trimming(eeg9, h9)
eeg9 -= np.mean(eeg9)
eeg9 /= np.std(eeg9)







data = pyedflib.EdfReader(Directory+"SC4211E0-PSG.edf")
eeg10=np.float32(data.readSignal(0))
del data
h10 = edfxRead(Directory+"SC4211EC-Hypnogram.edf")
eeg10 = eeg10[:np.uint(len(h10)*30*100)]
eeg10, h10 = trimming(eeg10, h10)
eeg10 -= np.mean(eeg10)
eeg10 /= np.std(eeg10)




data = pyedflib.EdfReader(Directory+"ST7022J0-PSG.edf")
eeg11=np.float32(data.readSignal(0))
del data
h11 = edfxRead(Directory+"ST7022JM-Hypnogram.edf")
eeg11 = eeg11[:np.uint(len(h11)*30*100)]
eeg11, h11 = trimming(eeg11, h11)
eeg11 -= np.mean(eeg11)
eeg11 /= np.std(eeg11)





data = pyedflib.EdfReader(Directory+"ST7061J0-PSG.edf")
eeg12=np.float32(data.readSignal(0))
del data
h12 = edfxRead(Directory+"ST7061JR-Hypnogram.edf")
eeg12 = eeg12[:np.uint(len(h12)*30*100)]
eeg12, h12 = trimming(eeg12, h12)
eeg12 -= np.mean(eeg12)
eeg12 /= np.std(eeg12)




data = pyedflib.EdfReader(Directory+"ST7121J0-PSG.edf")
eeg13=np.float32(data.readSignal(0))
del data
h13= edfxRead(Directory+"ST7121JE-Hypnogram.edf")
eeg13 = eeg13[:np.uint(len(h13)*30*100)]
eeg13, h13 = trimming(eeg13, h13)
eeg13 -= np.mean(eeg13)
eeg13 /= np.std(eeg13)




data = pyedflib.EdfReader(Directory+"ST7081J0-PSG.edf")
eeg14=np.float32(data.readSignal(0))
del data
h14 = edfxRead(Directory+"ST7081JW-Hypnogram.edf")
eeg14 = eeg14[:np.uint(len(h14)*30*100)]
eeg14, h14 = trimming(eeg14, h14)
eeg14 -= np.mean(eeg14)
eeg14 /= np.std(eeg14)




data = pyedflib.EdfReader(Directory+"ST7041J0-PSG.edf")
eeg15=np.float32(data.readSignal(0))
del data
h15 = edfxRead(Directory+"ST7041JO-Hypnogram.edf")
eeg15 = eeg15[:np.uint(len(h15)*30*100)]
eeg15, h15 = trimming(eeg15, h15)
eeg15 -= np.mean(eeg15)
eeg15 /= np.std(eeg15)




data = pyedflib.EdfReader(Directory+"ST7111J0-PSG.edf")
eeg16=np.float32(data.readSignal(0))
#eog16=np.float32(data.readSignal(2))
del data
h16 = edfxRead(Directory+"ST7111JE-Hypnogram.edf")
eeg16 = eeg16[:np.uint(len(h16)*30*100)]
eeg16, h16 = trimming(eeg16, h16)
eeg16 -= np.mean(eeg16)
eeg16 /= np.std(eeg16)




data = pyedflib.EdfReader(Directory+"ST7072J0-PSG.edf")
eeg17=np.float32(data.readSignal(0))
del data
h17 = edfxRead(Directory+"ST7072JA-Hypnogram.edf")
eeg17 = eeg17[:np.uint(len(h17)*30*100)]
eeg17, h17 = trimming(eeg17, h17)
eeg17 -= np.mean(eeg17)
eeg17 /= np.std(eeg17)





data = pyedflib.EdfReader(Directory+"ST7151J0-PSG.edf")
eeg18=np.float32(data.readSignal(0))
del data
h18 = edfxRead(Directory+"ST7151JA-Hypnogram.edf")
eeg18 = eeg18[:np.uint(len(h18)*30*100)]
eeg18, h18 = trimming(eeg18, h18)
eeg18 -= np.mean(eeg18)
eeg18 /= np.std(eeg18)







data = pyedflib.EdfReader(Directory+"ST7191J0-PSG.edf")
eeg19=np.float32(data.readSignal(0))
del data
h19 = edfxRead(Directory+"ST7191JR-Hypnogram.edf")
eeg19 = eeg19[:np.uint(len(h19)*30*100)]
eeg19, h19 = trimming(eeg19, h19)
eeg19 -= np.mean(eeg19)
eeg19 /= np.std(eeg19)



data = pyedflib.EdfReader(Directory+"ST7201J0-PSG.edf")
eeg20=np.float32(data.readSignal(0))
del data
h20 = edfxRead(Directory+"ST7201JO-Hypnogram.edf")
eeg20 = eeg20[:np.uint(len(h20)*30*100)]
eeg20, h20 = trimming(eeg20, h20)
eeg20 -= np.mean(eeg20)
eeg20 /= np.std(eeg20)






EEG = np.concatenate((eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, eeg8,
                      eeg9, eeg10, eeg11, eeg12, eeg13, eeg14, eeg15,
                      eeg16, eeg17, eeg18, eeg19, eeg20), axis=0)
              
H = np.concatenate((h1, h2, h3, h4, h5, h6, h7, h8, h9, h10,
                    h11, h12, h13, h14, h15, h16, h17, h18, h19,h20), axis=0)


n = 0
EEG1 = []
while (n < len(H)):
    
    EEG1.append(EEG[n*3000:(n+1)*3000])
    n += 1
     
EEG1 = np.array(EEG1)








