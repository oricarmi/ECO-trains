# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:46:16 2020

@author: Ori
"""

import pandas as pd
import numpy as np
from numpy import matlib as mtlb
import scipy as sp
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

def calcLAeq(spectrum):
    vec = [-70.4,-63.4,-56.7,-50.5,-44.7,-39.4,-34.6,-30.2,-26.2,-22.5,-19.1,-16.1,-13.4,-10.9,-8.6,-6.6,-4.8,-3.2,-1.9,-0.8,0.0,0.6,1.0,1.2,1.3,1.2,1.0,0.5,-0.1,-1.1,-2.5] #-4.3,-6.6]
    LAeq = np.sum(np.power((spectrum+vec)/10,10),axis=1)
    return [LAeq,10*np.log10(LAeq)]

def isContained(X,Y,windowSize):
    vec = np.in1d(X,Y) # WLOG: len(Y)>len(X), returns boolean of len(X) if element X is in Y
    return 1 if sum(vec)>0.6*windowSize else 0 # return true/false if at least 60% of the window is train 

def trainModel(Xtrain,Xtest,Ytrain,Ytest,What,toScale = 0):
    if toScale:
        scaler = StandardScaler() 
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)
    if What ==1: # ANN
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,3), random_state=1)
    else: # SVM
        clf = svm.SVC(C=5,kernel='rbf',class_weight={0:1,1:10})
    clf.fit(Xtrain, Ytrain) # fit the classifier to training data
    predictionTrain = clf.predict(Xtrain).astype(int) # predict on training data
    predictionTest = clf.predict(Xtest).astype(int) # predict on testing data
    resTest = dict() 
    resTest["confMat"] = confusion_matrix(Ytest, predictionTest)
    resTest["accuracy"] = np.trace(resTest["confMat"])/np.sum(resTest["confMat"])
    resTest["prediction"] = predictionTest
    resTrain = dict()
    resTrain["confMat"] = confusion_matrix(Ytrain, predictionTrain)
    resTrain["accuracy"] = np.trace(resTrain["confMat"])/np.sum(resTrain["confMat"])
    resTrain["prediction"] = predictionTrain
    return [resTest,resTrain]  # return results of train and test
      

freqs = [10,12.5,16,20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000];

#################################### YAHEL ####################################     
#df = pd.read_csv(r"C:\Users\Ori\Desktop\Ori\ECO\matlab\BS_csv.csv") # read csv flie
#dataY = df.to_numpy() 
#spectrumY = dataY[:,5:-3].astype("float") # spectrum from 10 Hz to 16,000 Hz
#TrainTimesY = [380,577,696,2822,3105,3756,4126,4126,6343,6573,7364,7692,7692,8429,9661,9973,10986,11296,11605]
#NorS = [2,1,2,1,1,2,2,1,1,2,2,2,1,1,2,1,2,1,1] # 2 - going south (from north) , 1 - going north (from south)
#t = dataY[:,1]
#LAeq,LAeq_dB = calcLAeq(spectrumY) # calc LAeq
#plt.figure
#plt.plot(LAeq)
#plt.figure
#plt.imshow(np.flipud(np.transpose(spectrumY)),extent=[0,14000,0,5000])
#TrainTimesInd = []
#for i in np.arange(len(NorS)): # iterate trains and specity times of train according to north/south
#    if NorS[i] == 2:
#        TrainTimesInd.append(np.arange(TrainTimesY[i]-30,TrainTimesY[i]+20))
#    else:
#        TrainTimesInd.append(np.arange(TrainTimesY[i]+30,TrainTimesY[i]+80))
#windowSize = 60
#increment = 1
#startPoints = np.arange(0,t.size-windowSize,increment) 
#windows = np.transpose(mtlb.repmat(startPoints,windowSize,1)) + mtlb.repmat(np.arange(windowSize),startPoints.size,1) # create windows of size windowSize and incrememnts of increment
#LAeq_windowed = LAeq[windows]
#spectrumWindowed = [] # start list of spectrum windowed (each item in list is the windowed frequency)
#for k in np.arange(spectrumY.shape[1]):
#    temp = spectrumY[:,k] # get this frequency
#    spectrumWindowed.append(temp[windows]) # window it and append to list 
#labelsY = np.zeros((windows.shape[0])) # start labels list
#for i in np.arange(len(TrainTimesInd)):
#    labelsY[np.round(TrainTimesInd[i][1] -0.4*windowSize).astype('int'):np.round(TrainTimesInd[i][-1]-0.6*windowSize).astype('int')] = 1
#FeaturesY = np.transpose(np.mean(np.asarray(spectrumWindowed),axis=2)) # features are mean spectrum of each window. rows - samples (# windows), cols - features (# frequencies)
#p = np.random.permutation(Features.shape[0]) # random permutation of samples
#Features2 = Features[p,:] # randomize the features
#labels2 = np.asarray(labels).astype(int) # make labels from boolean to [0,1]
#labels2 = labels2[p] # randomize the labels
#Xtrain = Features2[0:9718,:] # 75% data for training
#Ytrain = labels2[0:9718] # 75% labels for training 
#Xtest = Features2[9719:,:] # 25% labels for testing
#Ytest = labels2[9719:] # 25% labels for testing
#resTest,resTrain = trainModel(Xtrain,Xtest,Ytrain,Ytest,2,1) # train model: 1 - ANN, else- SVM, scale [1/0]
############################### END YAHEL #####################################
 
###################################### ORI ####################################     
df = pd.read_csv(r"C:\Users\Ori\Desktop\Ori\ECO\trains - python\trains beersheva ori.csv") # read csv flie
data = df.to_numpy() 
t = data[:,1]
spectrum = data[:,6:-5].astype("float") # spectrum from 10 Hz to 16,000 Hz   
#LAeq,LAeq_dB = calcLAeq(spectrum) # calc LAeq # if don't have laeq
LAeq_dB = data[:,3].astype("float")
LAeq = np.power(LAeq_dB/10,10).astype("float")
plt.figure
plt.plot(LAeq)
plt.figure
plt.imshow(np.flipud(np.transpose(spectrum)),extent=[0,14000,0,5000])
startInd = data[:,-2].astype("float")
startInd = startInd[~np.isnan(startInd)]
endInd = data[:,-1].astype("float")
endInd = endInd[~np.isnan(endInd)]
windowSize = 60
increment = 1
startPoints = np.arange(0,t.size-windowSize,increment) 
windows = np.transpose(mtlb.repmat(startPoints,windowSize,1)) + mtlb.repmat(np.arange(windowSize),startPoints.size,1) # create windows of size windowSize and incrememnts of increment
LAeq_windowed = LAeq[windows]
spectrumWindowed = [] # start list of spectrum windowed (each item in list is the windowed frequency)
for k in np.arange(spectrum.shape[1]):
    temp = spectrum[:,k] # get this frequency
    spectrumWindowed.append(temp[windows]) # window it and append to list 
labels = np.zeros((windows.shape[0])) # start labels list
for i in np.arange(startInd.shape[0]):
    labels[np.round(startInd[i] -0.4*windowSize).astype('int'):np.round(endInd[i]-0.6*windowSize).astype('int')] = 1
Features = np.transpose(np.mean(np.asarray(spectrumWindowed),axis=2)) # features are mean spectrum of each window. rows - samples (# windows), cols - features (# frequencies)
p = np.random.permutation(Features.shape[0]) # random permutation of samples
Features2 = Features[p,:] # randomize the features
labels2 = labels[p] # randomize the labels
Xtrain = Features2[0:round(Features.shape[0]*0.75),:] # 75% data for training
Ytrain = labels2[0:round(Features.shape[0]*0.75)] # 75% labels for training 
Xtest = Features2[round(Features.shape[0]*0.75)+1:,:] # 25% labels for testing
Ytest = labels2[round(Features.shape[0]*0.75)+1:] # 25% labels for testing
resTest,resTrain = trainModel(Xtrain,Xtest,Ytrain,Ytest,2,1) # train model: 1 - ANN, else- SVM, scale [1/0]
predTrainSorted = resTrain["prediction"][np.argsort(p)]
#resTest,resTrain = trainModel(Features2,FeaturesY,labels2,labelsY,2,1) # train model: 1 - ANN, else- SVM, scale [1/0]
