
from functions import *



Directory = "DREAMS_DatabaseSubjects\\"

# loadding DREAMS Subject dataset
EEG = []
H = []
for i in range(1,21):
    
    data = pyedflib.EdfReader(Directory + "subject"+str(i)+".edf")
    eeg = data.readSignal(2)
    h = loadtxt(Directory + "HypnogramAASM_subject"+str(i)+".txt",skiprows=1, comments="#", delimiter="\n", unpack=False)
    h = HYPNO(h)
    #dowsamplig signals to fs = 100Hz
    eeg = scipy.signal.resample_poly(eeg, up =100,  down = 200)
    eeg =  eeg[:len(h)*3000]
    #centering 
    eeg = eeg - np.mean(eeg)
    eeg /= np.std(eeg)

    EEG.extend(eeg)
    H.extend(h)
    print(i)

#setting up hypnogram as 30s labels
n = 0
EEG1 = []
while (n < len(H)):
    
    EEG1.append(EEG[n*3000:(n+1)*3000])
    n += 1
    
EEG1 = np.array(EEG1)
H=np.array(H)
    
    
    
    

