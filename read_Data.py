# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:17:43 2020

@author: Fabian
"""
#
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import h5py
from sklearn.metrics import f1_score

filename = "../../../media/user_home2/vision2020_01/Data/Los_cinefilos/multimodal_imdb.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[7]
    a_group_key_2 = list(f.keys())[1]

    # Get the data
    data = list(f[a_group_key])
    data2= list(f[a_group_key_2])
    
features=np.array(data[0:5000])
labels=np.array(data2[0:5000])
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X_train, y_train)



data=0
data2=0
#metrics 
Q=len(labels[0])
predecidos=clf.predict(X_test)


f1_weighted=f1_score(y_test,predecidos,average='weighted')
f1_macro=f1_score(y_test,predecidos,average='macro')
f1_micro=f1_score(y_test,predecidos,average='micro')
f1_samples=f1_score(y_test,predecidos,average='samples')

print('f1_weighted= '+str(f1_weighted))
print('f1_macro= '+str(f1_macro))
print('f1_micro= '+str(f1_micro))
print('f1_samples= '+str(f1_samples))

    





