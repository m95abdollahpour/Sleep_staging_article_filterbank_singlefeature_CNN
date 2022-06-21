
from functions import *

x_train, x_test, y_train, y_test = train_test_split(X, H, test_size=0.5,shuffle=True,
                                                        random_state=3)


clf1 = LinearDiscriminantAnalysis()   
clf1.fit(x_train,y_train)
a1=clf1.score(x_test, y_test)
y_pred = clf1.predict(x_test)
acc = clf1.score(x_test, y_test)
C1 = confusion_matrix(y_test, y_pred)
C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))
print('lda:\n', C11)


clf=RandomForestClassifier()
clf.fit(x_train,y_train)
a=clf.score(x_test, y_test)
y_pred = clf.predict(x_test)
C1 = confusion_matrix(y_test, y_pred)
C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))
print('RF:\n',C11)



neigh = KNeighborsClassifier(n_neighbors=8,algorithm='kd_tree',)
neigh.fit(x_train,y_train) 
a2=neigh.score(x_test, y_test)
y_pred = neigh.predict(x_test)
C1 = confusion_matrix(y_test, y_pred)
C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))
print('knn:\n',C11)



LR = LogisticRegression(random_state=0)
LR.fit(x_train,y_train)
a3=LR.score(x_test, y_test)
y_pred = LR.predict(x_test)
C1 = confusion_matrix(y_test, y_pred)
C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))
print('LR:\n',C11)


DT = tree.DecisionTreeClassifier()
DT = clf.fit(x_train,y_train)
a3=DT.score(x_test, y_test)
y_pred = DT.predict(x_test)
C1 = confusion_matrix(y_test, y_pred)
C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))
print('DT:\n',C11)


SVM = svm.SVC()
SVM.fit(x_train,y_train)
a3=SVM.score(x_test, y_test)
y_pred =SVM.predict(x_test)
C1 = confusion_matrix(y_test, y_pred)
C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))
print('SVM:\n',C11)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
a3=gnb.score(x_test, y_test)
y_pred =gnb.predict(x_test)
C1 = confusion_matrix(y_test, y_pred)
C11 =np.float16(100*( C1.astype('float') / C1.sum(axis=1)[:, np.newaxis]))
print('GNB:\n',C11)

