
# coding: utf-8

# In[1]:

import os, csv
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, decomposition, manifold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, train_test_split
from scipy.stats import randint as sp_randint
from time import time
from sklearn.decomposition import PCA, KernelPCA
import logging
import matplotlib.pyplot as plt
import pickle
import cv2
from scipy.misc import imresize
from skimage import feature
np.random.seed(1)


# In[2]:

#Load all the data and print keys
def sizeof(data):
    sz = len(data['subjectLabels'][:])
    return sz
# Load Data

#data = pickle.load(open("C:/CAS/UIE/Data/data-demo.pkl", 'rb'))
#data = pickle.load(open("E:/Data/a1_dataTrain.pkl",'rb'))
#print("loaded demo data")
# Note that the data is a dictionary with the following fields (where the names are self-explanatory):



# You may find other feature extraction techniques on: 
# http://scikit-image.org/docs/dev/api/skimage.feature.html
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html

def visualizeHOG(img, orientations=12, pixels_per_cell=(16,16), cells_per_block=(4,4),widthPadding=10, plotTitle=None):
    """
    Calculates HOG feature vector for the given image.
    
    img is a numpy array of 2- or 3-dimensional image (i.e., grayscale or rgb). 
    Color-images are first transformed to grayscale since HOG requires grayscale 
    images.
    
    Reference: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
    """
    if len(img.shape) > 2:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    # Crop the image from left and right.
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    # Note that we are using skimage.feature.
    hog_features, hog_image = feature.hog(img, orientations, pixels_per_cell, cells_per_block, visualise=True)
    
    # In order to visualize the result, you need to pass a plotTitle (you can remove the condition).
    if plotTitle is not None:
        plt.figure()
        plt.suptitle(plotTitle)
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(hog_image)
        plt.axis("off")
        
        print("[" + plotTitle + "] # Features: " + str(hog_features.shape))
    return hog_features



# In[ ]:

def visualizeSIFT(img, contrastThreshold=0.04, edgeThreshold=10, widthPadding=10, plotTitle=None):
    """
    Calculates SIFT feature vector for the given image. Note that SIFT operation 
    detects arbitrary keypoints for different images.

    img is a numpy array of 2- or 3-dimensional image (i.e., grayscale or rgb). 

    Since SIFT can extract different # of key points per image, you can use nfeatures 
    parameter of the SIFT create() function in order to control it. 
    
    For details:
        [nfeatures]:The number of best features to retain. The features are
        ranked by their scores (measured in SIFT algorithm as the local contrast)
        [contrastThreshold]:The contrast threshold used to filter out weak features
        in semi-uniform (low-contrast) regions. The larger the threshold, the less
        features are produced by the detector.
        [edgeThreshold]:The threshold used to filter out edge-like features. Note
        that the its meaning is different from the contrastThreshold, i.e. the
        larger the edgeThreshold, the less features are filtered out (more
        features are retained).
        
    A tutorial: http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    """
    # Crop the image from left and right.
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    # Create sift object.
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
    
    # detectAndCompute method first detects the keypoints and then calculates description vectors.
    # You may first use detect, and then compute methods as well. It is useful if you want to 
    # detect the keypoints for an image and then use the same keypoints on other images.
    (kp, descriptions) = sift.detectAndCompute(img, None)
    SIFT_features=descriptions
    # In order to visualize the result, you need to pass a plotTitle (you can remove the condition).
    if plotTitle is not None:
        # For visualization
        imgSift = np.copy(img)
        imgSift=cv2.drawKeypoints(img,kp,imgSift,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure()
        plt.suptitle(plotTitle)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(imgSift)
        plt.axis("off")
        
        print("[" + plotTitle + "] # Features: " + str(descriptions.size))
    return SIFT_features   


# In[ ]:

def visualizeSIFTDense(img, pixelStepSize=10, widthPadding=10, plotTitle=None):
    """
    Calculates SIFT feature vector for the given image. First, a grid of keypoints 
    is created by scanning through the pixel locations of the image.
    Note that if you use the same grid for every image, then features will have the
    same locality for other images as well.

    img is a numpy array of 2- or 3-dimensional image (i.e., grayscale or rgb).
    """
    # Crop the image from left and right.
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]
    
    # Create sift object.
    sift = cv2.xfeatures2d.SIFT_create()
    
    # Create grid of key points.
    keypointGrid = [cv2.KeyPoint(x, y, pixelStepSize)
                    for y in range(0, img.shape[0], pixelStepSize)
                        for x in range(0, img.shape[1], pixelStepSize)]
    
    # Given the list of keypoints, compute the local descriptions for every keypoint.
    (kp, descriptions) = sift.compute(img, keypointGrid)
    denseSIFT_features = descriptions
    if plotTitle is not None:    
        # For visualization
        imgSiftDense = np.copy(img)
        imgSiftDense=cv2.drawKeypoints(img,kp,imgSiftDense,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure()
        plt.suptitle(plotTitle)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(imgSiftDense)
        plt.axis("off")
        
        print("[" + plotTitle + "] # Features: " + str(descriptions.size))
    return denseSIFT_features


# In[7]:

def computeFeatures(data):
# Compute Features
    X=[]
    print("computing some features")
    for sampleIdx in range(sizeof(data)):
    #compute masks
        segmentedUser = data['segmentation'][sampleIdx]
        mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
        mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
        mask3 = mask3.transpose((1,2,0))
    
       # Masked Depth
        maskedDepth = data['depth'][sampleIdx] * mask2
        denseSIFT = visualizeSIFTDense(maskedDepth)
        dephog  = visualizeHOG(maskedDepth)
        # Masked RGB
        maskedImg = data['rgb'][sampleIdx]* mask3
        hog_features=visualizeHOG(maskedImg)
        #SIFT_features=visualizeSIFT(maskedDepth)
 
        # Segmentation Mask Modality
        sourceImg = data['segmentation'][sampleIdx]
        seghog_features = visualizeHOG(sourceImg)
        #segSIFT_features=visualizeSIFT(sourceImg)
    
    
        # Masked Depth
        maskedDepth = data['depth'][sampleIdx] * mask2
        dephog  = visualizeHOG(maskedDepth)
        #denseSIFT = visualizeSIFTDense(maskedDepth)
        # Actual Features used
        #features=hog_features
        features = hog_features
        features = np.concatenate((hog_features, seghog_features, dephog),axis=0)
        X.append(features)
    X=np.asarray(X)
    print(str(X.shape))    
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(X)
    return X_kpca


# In[ ]:

#X=computeFeatures(data)
#y=data['gestureLabels'][:]

#Split into train and test sets This is given for actual data and therefore not needed and it is read from the files

#X_train, X_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=1)


#print(str(X_train.shape) + " - " + str(X_test.shape))

# print(labels_test)


# In[8]:

Path = "Data/"
filelist = os.listdir(Path)
print(filelist)


# In[9]:

# Use actual data
TrainData = pickle.load(open("a1_dataTrain.pkl",'rb'))
labels_train = TrainData['gestureLabels'][:]
X_train      = computeFeatures(TrainData)
print("Done Training")
TestData = pickle.load(open("a1_dataTest.pkl",'rb'))
X_test = computeFeatures(TestData)
print(str(X_train.shape) + " - " + str(X_test.shape))
print(TestData.keys())
labels_test= []
print("yipppee")
#print(str(X_train.shape) + " - " + str(X_test.shape))

# print(labels_test)


# In[ ]:




# Random forest
from sklearn.model_selection import KFold, PredefinedSplit, ShuffleSplit


# In[ ]:

# cv parameter of RandomizedSearchCV or GridSearchCV can be fed with a customized cross-validation object.
ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
                  
# Optimize the parameters by cross-validation.
parameters = {
        "max_depth": sp_randint(15, 25),
        "max_features": sp_randint(16, 40),
        "min_samples_split": sp_randint(20, 40),
        "min_samples_leaf": sp_randint(10, 25),
        'n_estimators': [10,20,30,40],
    }

# Random search object with SVM classifier.
clf = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=1),
        param_distributions=parameters,
        n_iter=10,
        cv=10,
        random_state=1,
    )
clf.fit(X_train, labels_train)

print("Best parameters set found on training set:")
print(clf.best_params_)
print()

means_valid = clf.cv_results_['mean_test_score']
stds_valid = clf.cv_results_['std_test_score']
means_train = clf.cv_results_['mean_train_score']

print("Grid scores:")
for mean_valid, std_valid, mean_train, params in zip(means_valid, stds_valid, means_train, clf.cv_results_['params']):
    print("Validation: %0.3f (+/-%0.03f), Training: %0.3f  for %r" % (mean_valid, std_valid, mean_train, params))
print()

labels_test, labels_predicted = labels_test, clf.predict(X_test)





#Write test results
print(len(labels_predicted)) 

with open('ResRf.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    header = ['Id','Prediction']
    csvwriter.writerow(header)
    for Idx in range(len(labels_predicted)):
        row =[str(Idx+1),str(labels_predicted[Idx])]
        csvwriter.writerow(row)
print(os.getcwd())


# In[ ]:



