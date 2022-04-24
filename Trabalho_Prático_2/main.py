"""
    Membros:
     - Carlos Jordão
     - Francisco Carreira
     - Gonçalo Folhas
"""


from fileinput import filename
from tkinter import N
from cv2 import norm
import librosa
import librosa.display
import scipy
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

    
def showFeats(file, array):
    array = np.genfromtxt(file, delimiter=',')
    nl, nc = array.shape
    print("dim ficheiro top100_features_normalized_data.csv = ", nl, "x", nc)
    print()
    print(array)



def extractFeatures(file):

    list_of_files = os.listdir(file)
    num_of_files = len(list_of_files)
    features = np.empty((num_of_files, 190))

    for i in range(num_of_files):
        print(f"File {i + 1} of {num_of_files}")
        fp = os.path.join(file, list_of_files[i])
        samples, sample_rate = librosa.load(fp)

        mfcc = librosa.feature.mfcc(y=samples, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=samples)
        spectral_bandwith = librosa.feature.spectral_bandwidth(y=samples)
        spectral_contrast = librosa.feature.spectral_contrast(y=samples)
        spectral_flatness = librosa.feature.spectral_flatness(y=samples)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=samples)
        fundamental = librosa.yin(y=samples, fmin=20, fmax=sample_rate/2)
        fundamental[fundamental == sample_rate/2] = 0
        rms = librosa.feature.rms(y=samples)
        zcr = librosa.feature.zero_crossing_rate(y=samples)
        tempo = librosa.beat.tempo(y=samples)

        features[i, : 91] = typicalStatistics(mfcc)
        features[i, 91 : 98] = typicalStatistics(spectral_centroid)
        features[i, 98 : 105] = typicalStatistics(spectral_bandwith)
        features[i, 105 : 154] = typicalStatistics(spectral_contrast)
        features[i, 154 : 161] = typicalStatistics(spectral_flatness)
        features[i, 161 : 168] = typicalStatistics(spectral_rolloff)
        features[i, 168 : 175] = typicalStatistics(fundamental)
        features[i, 175 : 182] = typicalStatistics(rms)
        features[i, 182 : 189] = typicalStatistics(zcr)
        features[i, 189] = tempo

    return features


def typicalStatistics(array):
    axis = 1 if array.ndim > 1 else 0
    mean = np.mean(array, axis=axis)
    standard_deviation = np.std(array, axis=axis)
    skewness = scipy.stats.skew(array, axis=axis)
    kurtosis = scipy.stats.kurtosis(array, axis=axis)
    median = np.median(array, axis=axis)
    max = np.max(array, axis=axis)
    min = np.min(array, axis=axis)

    if axis == 0:
        array = np.array([mean, standard_deviation, skewness, kurtosis, median, max, min])
    else:
        array = np.empty((array.shape[0], 7))
        for i in range(array.shape[0]):
            array[i, :] = [mean[i], standard_deviation[i], skewness[i], kurtosis[i], median[i], max[i], min[i]]

    return array.flatten()


def normalization(fts):
    for i in range(fts.shape[1]):
        vmax = np.max(fts[:, i])
        vmin = np.min(fts[:, i])
        fts[:, i] = (fts[:, i] - vmin) / (vmax - vmin)

    return fts





if __name__ == "__main__":

    #2.1
    """ fileName = './Features - Audio MER/top100_features.csv'
    feats = featureArr(fileName)
    normalizedFeats = normalizeFeatures(feats)
    fileToSave = './Features - Audio MER/top100_normalized_features.csv'
    saveFeats(fileToSave, normalizedFeats)
    showFeats(fileToSave, normalizedFeats) """

    #2.2
    features = extractFeatures("./MER_audio_taffc_dataset/all")
    features = normalization(features)
    fileToSave = './Features - Audio MER/900audios_normalized_features.csv'
    saveFeats(fileToSave, features)