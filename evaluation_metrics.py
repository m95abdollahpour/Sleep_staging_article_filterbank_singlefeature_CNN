from functions import *


m5cls = load_model('two_stage_cnn.h5')



x_train, x_test, y_train, y_test = train_test_split(X_edf, H, test_size=0.2,
                                                    shuffle=True,random_state=3)

x_train = np.reshape(x_train, (x_train.shape[0],  x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0],  x_test.shape[1], 1))

y_pred = m5cls.predict([x_test[:,:49,:],x_test[:,49:,:]])

ypred= np.argmax(y_pred, axis=1)



C1 = confusion_matrix(y_test, ypred)

C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))


kappa6 = sklearn.metrics.cohen_kappa_score(y_test, ypred)



cnf_matrix = C1

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
acc= np.sum(TP)/(TP+FP+FN+TN)[0]

SENS = TP / (TP + FN)
SPEC = TN / (TN + FP)
F1 = 2*TP/(2*TP+FP+FN)
PREC=TP/(TP+FP)




