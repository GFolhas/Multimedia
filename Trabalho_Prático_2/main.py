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
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st
import scipy.spatial.distance as ssd
import pandas
from os import listdir, makedirs
from os.path import isfile, isdir



def featureArr(file):
    print(" Reading Top 100 features file")
    top100 = np.genfromtxt(file, delimiter=',')
    nl, nc = top100.shape
    print("dim ficheiro top100_features.csv original = ", nl, "x", nc)
    print()
    print(top100)
    top100 = top100[1:, 1:(nc-1)] #eliminar a 1ª linha e 1ª coluna
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
        f0 = librosa.yin(y=samples, fmin=20, fmax=sample_rate/2)
        f0[f0 == sample_rate/2] = 0
        rms = librosa.feature.rms(y=samples)
        zcr = librosa.feature.zero_crossing_rate(y=samples)
        tempo = librosa.beat.tempo(y=samples)

        features[i, : 91] = typicalStatistics(mfcc)
        features[i, 91 : 98] = typicalStatistics(spectral_centroid)
        features[i, 98 : 105] = typicalStatistics(spectral_bandwith)
        features[i, 105 : 154] = typicalStatistics(spectral_contrast)
        features[i, 154 : 161] = typicalStatistics(spectral_flatness)
        features[i, 161 : 168] = typicalStatistics(spectral_rolloff)
        features[i, 168 : 175] = typicalStatistics(f0)
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

    if axis != 0:
        array = np.empty((array.shape[0], 7))
        for i in range(array.shape[0]):
            array[i, :] = [mean[i], standard_deviation[i], skewness[i], kurtosis[i], median[i], max[i], min[i]]
    else:
        array = np.array([mean, standard_deviation, skewness, kurtosis, median, max, min])
    return array.flatten()



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
            #print(f"({n}, {m})")
            top100Euclidean[n, m] = ssd.euclidean(top100[n, :], top100[m, :])
            featuresEuclidean[n, m] = ssd.euclidean(features[n, :], features[m, :])
            top100Manhattan[n, m] = ssd.cityblock(top100[n, :], top100[m, :])
            featuresManhattan[n, m] = ssd.cityblock(features[n, :], features[m, :])
            top100Cosine[n, m] = ssd.cosine(top100[n, :], top100[m, :])
            featuresCosine[n, m] = ssd.cosine(features[n, :], features[m, :])


    distances = {
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

    np.savetxt("dist/top100/euclidean.csv", distances['top100']['euclidean'], delimiter=",", fmt="%f")
    np.savetxt("dist/top100/manhattan.csv", distances['top100']['manhattan'], delimiter=",", fmt = "%f")
    np.savetxt("dist/top100/cosine.csv", distances['top100']['cosine'], delimiter=",", fmt = "%f")
    np.savetxt("dist/features/euclidean.csv", distances['features']['euclidean'], delimiter=",", fmt = "%f")
    np.savetxt("dist/features/manhattan.csv", distances['features']['manhattan'], delimiter=",", fmt = "%f")
    np.savetxt("dist/features/cosine.csv", distances['features']['cosine'], delimiter=",", fmt = "%f")
    return [[top100Euclidean, top100Manhattan, top100Cosine], [featuresEuclidean, featuresManhattan, featuresCosine]]


def normalization(fts):
    for i in range(fts.shape[1]):
        vmax = np.max(fts[:, i])
        vmin = np.min(fts[:, i])
        fts[:, i] = (fts[:, i] - vmin) / (vmax - vmin)

    return fts



def ranking(index, matrix, dset, path):
    dist = ['Euclidean', 'Manhattan', 'Cosine']
    ranking = [[], [], []]
    for i in range(3):
        row = matrix[i][index, :]
        indices = np.argsort(row)[1:21]
        for j in indices:
            ranking[i].append(dset[j].split("/")[-1])

        if not isdir(path): makedirs(path)
        #file = path + dist[i] + ".txt"
        file = f"{path}/{dist[i]}.txt"
        if isfile(file): continue
        print("here")
        print(file)
        with open(file, "w") as file:
            for song in ranking[i]:
                print(song, file=file)

        """ if path is not None:
            if not isdir(path): makedirs(path)
            filename = f"{path}/{dset[index].split('/')[-1]}.txt"
            if not isfile(filename):
                with open(filename, 'w') as file:
                    for song in ranking:
                        print(song, file=file) """

    return ranking




def mdScores(index, metadata, size):
    scores = np.zeros((1, size))
    scores[0, index] = -1

    for i in range(len(metadata)):
        if i != index:
            score = 0

            if metadata['Artist'][i] == metadata['Artist'][index]: score += 1
            if metadata['Quadrant'][i] == metadata['Quadrant'][index]: score += 1

            for j in metadata['GenresStr'][i].split("; "):
                for k in metadata['GenresStr'][index].split("; "):
                    if j == k:
                        score += 1
                        break

            for j in metadata['MoodsStrSplit'][i].split("; "):
                for k in metadata['MoodsStrSplit'][index].split("; "):
                    if j == k:
                        score += 1
                        break

            scores[0, i] = score

    return scores


def mdRanking(index, metadata, dset, path):
    scores = mdScores(index, metadata, len(metadata))[0]

    indices = np.argsort(scores)[::-1][:20]

    ranking = list()
    for i in indices:
        print(i)
        ranking.append(dset[i].split("/")[-1])

    if path is not None:
        if not isdir(path): makedirs(path)
        filename = f"{path}/{dset[index].split('/')[-1]}.txt"
        if not isfile(filename):
            with open(filename, 'w') as file:
                for song in ranking:
                    print(song, file=file)

    return ranking


def precision(feat, t100, md):
    mds = set(md)
    pr = [[i(0, feat, md, mds), i(1, feat, md, mds), i(2, feat, md, mds)],[i(0, t100, md, mds), i(1, t100, md, mds), i(2, t100, md, mds)]]
    return pr

def i(int, arr, arr2, sets):
    return len(set(arr[0]).intersection(sets)) / len(arr2) * 100;




if __name__ == "__main__":

    #2.1
    """ fileName = './Features - Audio MER/top100_features.csv'
    feats = featureArr(fileName)
    normalizedFeats = normalizeFeatures(feats)
    fileToSave = './Features - Audio MER/top100_normalized_features.csv'
    saveFeats(fileToSave, normalizedFeats)
    showFeats(fileToSave, normalizedFeats)

    #2.2
    features = extractFeatures("./MER_audio_taffc_dataset/all")
    features = normalization(features)
    fileToSave = './Features - Audio MER/900audios_normalized_features.csv'
    saveFeats(fileToSave, features) 
    """

    #3
    top100: np.ndarray
    if os.path.isfile("Features - Audio MER/top100.csv"):
        top100 = np.genfromtxt("Features - Audio MER/top100.csv", delimiter=",")
    else:
        top100 = featureArr("Features - Audio MER/top100_features.csv")
        np.savetxt("Features - Audio MER/top100.csv", top100, fmt="%f", delimiter=",")
    features: np.ndarray
    if os.path.isfile("Features - Audio MER/900audios_normalized_features.csv"):
        features = np.genfromtxt("Features - Audio MER/900audios_normalized_features.csv", delimiter=",")
    else:
        features = extractFeatures("dataset/all")
        np.savetxt("Features - Audio MER/900audios_normalized_features.csv", features, fmt="%f", delimiter=",")
    distances = calculateDistances(top100, features)
    d_top100 = distances[0]
    d_features = distances[1]

    #4.1
    # Read metadata csv
    p = "./MER_audio_taffc_dataset/all"
    dset = [f"{p}/{x}" for x in sorted(listdir(p))]
    fileName = "./MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv"
    queries = listdir("./Queries")

    #verify and print all the queries
    for query in queries:
        print("Query: ", query)
        index = dset.index(f"{p}/{query}")
        featuresRanking = ranking(index, d_features, dset, path=f"featuresRankings")
        top100Ranking = ranking(index, d_top100, dset, path=f"top100Rankings")
        metadataCollumns = ['Artist', 'Song', 'GenresStr', 'Quadrant', 'MoodsStrSplit']
        metadata = pandas.read_csv(fileName, usecols=metadataCollumns)
        metadataRanking = mdRanking(index, metadata, dset, path=f"metadataRankings")
        precise = precision(featuresRanking, top100Ranking, metadataRanking)
        print("\nPrecisions: ")
        print(precise)