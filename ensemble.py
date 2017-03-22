
# coding: utf-8

# In[4]:

import os, csv
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, decomposition, manifold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, train_test_split
from scipy.stats import randint as sp_randint
from time import time
import logging
import matplotlib.pyplot as plt
import pickle
get_ipython().magic('matplotlib inline')
import cv2
from scipy.misc import imresize
from scipy import stats
from skimage import feature
np.random.seed(1)


# #  Calculate an ensemble of classifiers by computing the mode of the final results (voting)

# In[19]:

Path = "E:/csvdir/"
filelist = os.listdir(Path)
print(filelist)
list_ =[]
for file in filelist:
    a = pd.read_csv(Path+file, index_col=None, header=0)
    list_.append(np.asarray(a['Prediction']))
final = stats.mode(np.asarray(list_),axis=0)
labels_predicted = np.transpose(final.mode)
print(np.squeeze(labels_predicted))
os.chdir (Path)
with open('ResMode.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    header = ['Id','Prediction']
    csvwriter.writerow(header)
    for Idx in range(len(labels_predicted)):
        row =[str(Idx+1),np.squeeze(labels_predicted[Idx])]
        csvwriter.writerow(row)


# In[ ]:




# In[ ]:



