from fileinput import filename
import librosa
import librosa.display
import sounddevice as sd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st



def featureArr(file):
    print(" Reading Top 100 features file")
    top100 = np.genfromtxt(file, delimiter=',')
    nl, nc = top100.shape
    print("dim ficheiro top100_features.csv original = ", nl, "x", nc)
    print()
    print(top100)
    top100 = top100[1:, 1:(nc-1)]
    nl, nc = top100.shape
    print()
    print("dim top100 data = ", nl, "x", nc)
    print()
    print(top100)
    return top100


def normalizeFeatures(array):
    print(" Feature Normalization")
    t100 = np.zeros(array.shape)
    nl, nc = t100.shape
    for i in range(nc):
        vmax = array[:, i].max()
        vmin = array[:, i].min()
        t100[:, i] = (array[:, i] - vmin)/(vmax - vmin)
    print(t100)
    return t100

def saveFeats(file, array):
    print(" Saving normalized data file")
    np.savetxt(file, array, fmt = "%lf", delimiter=',')

    array = np.genfromtxt(file, delimiter=',')
    nl, nc = array.shape
    print("dim ficheiro top100_features_normalized_data.csv = ", nl, "x", nc)
    print()
    print(array)





if __name__ == "__main__":
    fileName = './Features - Audio MER/top100_features.csv'
    feats = featureArr(fileName)
    normalizedFeats = normalizeFeatures(feats)
    fileToSave = './Features - Audio MER/top100_normalized_features.csv'
    saveFeats(fileToSave, normalizedFeats)
