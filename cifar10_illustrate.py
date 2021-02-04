import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from random import random, randrange

#class accuracy calculator
def class_acc(pred, pred2, gt, totalPred, correctPred, correctPred2):
    totalPred += 1
    if(pred == gt):
        correctPred += 1
    if(pred2 == gt):
        correctPred2 += 1
    accRating = correctPred / totalPred * 100
    accRating2 = correctPred2 / totalPred * 100
    plt.text(5, -3,'Random prediction: ' + pred, ha="center", va="center")
    plt.text(25, -3,'1NN prediction: ' + pred2, ha="center", va="center")
    plt.text(37, 5,'Random\nClassifier\nCurrent\nAccuracy:\n ' + str(round(accRating, 1)) + '%' , ha="center", va="center")
    plt.text(37, 15,'1NN\nClassifier\nCurrent\nAccuracy:\n ' + str(round(accRating2, 1)) + '%' , ha="center", va="center")
    return correctPred, correctPred2, totalPred

#"random calssifier" returns a random label
def cifar10_classifier_random(x):
    return x[randrange(10)]

#one NN classifier
def cifar10_classifier_onenn(x, trdata, trlabels):
    labelnscore = 'none', 999999
    for i in range(trdata.shape[0]):
        if(np.sum(np.abs(x - trdata[i])) < labelnscore[1]):
            labelnscore = trlabels[Ya[i]], np.sum(np.abs(x - trdata[i]))
    return labelnscore[0]

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

testdict = unpickle('./cifar-10-batches-py/test_batch')
datadict = unpickle('./cifar-10-batches-py/data_batch_' + str(randrange(1, 6)))
datadict1 = unpickle('./cifar-10-batches-py/data_batch_1')
datadict2 = unpickle('./cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('./cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('./cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('./cifar-10-batches-py/data_batch_5')

X = datadict["data"]
Y = datadict["labels"]
X2 = testdict["data"]
Y2 = testdict["labels"]
Xa = datadict1["data"]
Ya = datadict1["labels"]
Xb = datadict2["data"]
Yb = datadict2["labels"]
Xc = datadict3["data"]
Yc = datadict3["labels"]
Xd = datadict4["data"]
Yd = datadict4["labels"]
Xe = datadict5["data"]
Ye = datadict5["labels"]

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int")
Y = np.array(Y)
X2 = X2.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int")
Y2 = np.array(Y2)
Xa = Xa.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int")
Ya = np.array(Ya)
Xb = Xb.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int")
Yb = np.array(Yb)
Xc = Xc.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int")
Yc = np.array(Yc)
Xd = Xd.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int")
Yd = np.array(Yd)
Xe = Xe.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("int")
Ye = np.array(Ye)

Xa = np.concatenate([Xa, Xb])
Xa = np.concatenate([Xa, Xc])
Xa = np.concatenate([Xa, Xd])
Xa = np.concatenate([Xa, Xe])

Ya = np.concatenate([Ya, Yb])
Ya = np.concatenate([Ya, Yc])
Ya = np.concatenate([Ya, Yd])
Ya = np.concatenate([Ya, Ye])
#amount of correct and total predictions
correctPred = 0
correctPred2 = 0
totalPred = 0

#main
for i in range(X.shape[0]):
    plt.figure(1)
    plt.clf()
    plt.imshow(X2[i])
    plt.text(2, 34,'images done: ' + str(i+1), ha="center", va="center")
    plt.title(f"Image {i} label={label_names[Y2[i]]} (num {Y2[i]})")
    first = time.time()
    correctPred, correctPred2, totalPred = class_acc(cifar10_classifier_random(label_names), cifar10_classifier_onenn(X2[i], Xa, label_names), label_names[Y2[i]], totalPred, correctPred, correctPred2)
    second = time.time()
    plt.text(20, 34,'computing time: ' + str(round(second - first, 3)) + 's', ha="center", va="center")
    plt.pause(0.1)
