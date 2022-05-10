"""
    Membros:
     - Carlos Jordão
     - Francisco Carreira
     - Gonçalo Folhas
"""

from genericpath import isdir, isfile
from os import listdir, makedirs
import os.path
import warnings
import numpy as np
import librosa
import scipy.stats
from scipy.spatial.distance import cityblock, euclidean, cosine
import pandas as pd


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

def ranking(idx: int, matrices: tuple, dataset: list, path: str) -> list:
    dist = ['euclidean', 'manhattan', 'cosine']

    ranking = [[], [], []]

    for i in range(3):
        if not isdir(path): makedirs(path)
        filename = f"{path}/{dist[i]}.txt"
        if isfile(filename): continue
        with open(filename, "w") as file:
            row = matrices[i][idx, :]
            indices = np.argsort(row)[1:21]
            for i in indices:
                ranking[i].append(dataset[i].split('/')[-1])
                print(dataset[i].split('/')[-1], file=file)

    return ranking

def normalizeFeatures(array):
    print("Feature Normalization")
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

def saveDistances(dist: dict):
    np.savetxt("distances/top100/euclidean.csv", dist['top100']['euclidean'], delimiter=",", fmt="%f")
    np.savetxt("distances/top100/manhattan.csv", dist['top100']['manhattan'], delimiter=",", fmt = "%f")
    np.savetxt("distances/top100/cosine.csv", dist['top100']['cosine'], delimiter=",", fmt = "%f")
    np.savetxt("distances/features/euclidean.csv", dist['features']['euclidean'], delimiter=",", fmt = "%f")
    np.savetxt("distances/features/manhattan.csv", dist['features']['manhattan'], delimiter=",", fmt = "%f")
    np.savetxt("distances/features/cosine.csv", dist['features']['cosine'], delimiter=",", fmt = "%f")

def calculateDistances(top100: np.ndarray, features: np.ndarray):
    features[np.isnan(features)] = 0
    top100Euclidean = np.empty((900, 900))
    top100Manhattan = np.empty((900, 900))
    top100Cosine = np.empty((900, 900))
    featuresEuclidean = np.empty((900, 900))
    featuresManhattan = np.empty((900, 900))
    featuresCosine = np.empty((900, 900))

    for n in range(top100.shape[0]):
        for m in range(top100.shape[0]):
            print(f"({n}, {m})")
            top100Euclidean[n, m] = euclidean(top100[n, :], top100[m, :])
            featuresEuclidean[n, m] = euclidean(features[n, :], features[m, :])
            top100Manhattan[n, m] = cityblock(top100[n, :], top100[m, :])
            featuresManhattan[n, m] = cityblock(features[n, :], features[m, :])
            top100Cosine[n, m] = cosine(top100[n, :], top100[m, :])
            featuresCosine[n, m] = cosine(features[n, :], features[m, :])

    return {
        'top100': {
            'euclidean': top100Euclidean,
            'manhattan': top100Manhattan,
            'cosine': top100Cosine
        },
        'features': {
            'euclidean': featuresEuclidean,
            'manhattan': featuresManhattan,
            'cosine': featuresCosine
        }
    }

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

    #3
    top100: np.ndarray
    if os.path.isfile("features/top100.csv"):
        top100 = np.genfromtxt("features/top100.csv", delimiter=",")
    else:
        top100 = featureArr("dataset/features/top100_features.csv")
        np.savetxt("features/top100.csv", top100, fmt="%f", delimiter=",")
    features: np.ndarray
    if os.path.isfile("features/librosa.csv"):
        features = np.genfromtxt("features/librosa.csv", delimiter=",")
    else:
        features = extractFeatures("dataset/all")
        np.savetxt("features/librosa.csv", features, fmt="%f", delimiter=",")
    distances = calculateDistances(top100, features)
    saveDistances(distances)

        # Read metadata csv
    metadataCols = ['Song', 'Artist', 'GenresStr', 'Quadrant', 'MoodsStrSplit']
    metadata = pd.read_csv("dataset/panda_dataset_taffc_metadata.csv", usecols=metadataCols)
