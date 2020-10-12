# sensitivity analysis of k in k-fold cross-validation
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('datasets_1846_3197_Social_Network_Ads-3.csv')
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values
# #Extract all row and columns 3 and 5
# X = dataset.iloc[:, [2, 3]].values
# #Extract "Purchased" values (1 if purchased, 0 if not)
# Y = dataset.iloc[:, 4].values
maxAvg = -1
resp=[]
k=range(1,11)
for i in k:
    mayor=False
    knn = KNeighborsClassifier(n_neighbors=i)
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
    knn.fit(x_train,y_train)
    kf = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(knn, x, y, scoring='accuracy', cv=kf, n_jobs=-1)
    # print(scores)
    resp.append([i,np.average(scores),False])
for value in resp:
    if(maxAvg<0 or maxAvg<=value[1]):
        maxAvg=value[1]
        value[2]=True
for value in resp:
    if(value[1]!=maxAvg):
        value[2]=False
print(resp)




# if(len(resp)>0):
#         for value in resp:
# 	        if(value[1]==maxAvg and np.average(scores)>=value[1]):
# 	    	    maxAvg=value[1]
# 	    	    mayor = True
# 	    	    value[2] = False
# 	    	    print("entre")
# 	    resp.append([i,np.mean(scores),mayor])
#     else:
#         maxAvg = np.average(scores)
#         resp.append([i,maxAvg,True])
