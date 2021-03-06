from os.path import isdir, basename, join, splitext
import os
import numpy
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
from glob import glob
import scipy.cluster.vq as vq
import cPickle
import pandas as pd
from descriptor.siftdesciptor import computeSIFT, computeHistograms
from descriptor.logger import Logger
import sys

datasetpath = 'dataset_img'
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
CODEBOOK_FILE = 'codebook.file'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
HISTOGRAMS_FILE = 'siftFeatures.csv'
TOTAL_FEATURES_FILE = 'total_features.file'


def get_categories(datasetpath):
    cat_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname))
                      for fname in glob(path + "/*")
                      if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files


def dict2numpy(dict, total_features):
    array = zeros((total_features, 128), dtype=numpy.float32)
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array


def extractSift(input_files, mask=False):
    all_features_dict = {}
    num_descriptor = 0
    for i, fname in enumerate(input_files):
        kp, des = computeSIFT(fname, mask)
        if (len(kp) == 0):
            print "No feature found for image :", fname
        else:
            num_descriptor += len(des)
        all_features_dict[fname] = des
    return all_features_dict, num_descriptor


if __name__ == '__main__':
    sys.stdout = Logger("log/kMean_log.txt")
    print "---------------------"
    print "## loading the dataset_img and extracting the sift features"
    cats = get_categories(datasetpath)
    ncats = len(cats)
    print "searching for folders at " + datasetpath
    print "found following folders / categories:"
    print cats
    print "---------------------"
    all_files = []
    all_files_labels = {}
    cat_label = {}
    cat_dic = {}
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_dic[cat] = cat_files
        all_files = all_files + cat_files
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label


    print "##Load all features from file"
    output_path = os.path.abspath("../app/training_data") + "/" + TOTAL_FEATURES_FILE
    print output_path
    with open(output_path, 'rb') as f:
        # save codebook into a binary file
        all_features=cPickle.load(f)


    total_descriptor=0;
    for k in all_features:
        total_descriptor=total_descriptor+len(all_features[k])

    print total_descriptor


    print "---------------------"
    print "## computing the visual words via k-means"
    all_features_array = dict2numpy(all_features, total_descriptor)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures / 2))
    nclusters = 100
    print "Number of cluster:", nclusters
    codebook, distortion = vq.kmeans(all_features_array,
                                     nclusters,
                                     thresh=K_THRESH)
    print "k-Means terminated."

    output_path = os.path.abspath("../app/training_data") + "/" + CODEBOOK_FILE
    with open(output_path, 'wb') as f:
        # save codebook into a binary file
        cPickle.dump(codebook, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print "## compute the visual words histograms for each image"
    print "Number of cluster: ", nclusters
    all_word_histgrams = {}
    columns = range(0, nclusters)
    columns.append('file_name')
    columns.append('flower_name')
    df = pd.DataFrame(columns=columns)

    for cat in cat_dic:
        for imagefname in cat_dic[cat]:
            word_histgram = computeHistograms(codebook, all_features[imagefname])
            # create dataframe to store histogram of visual word occurences
            # Convert feature vector to a list
            result = numpy.squeeze(word_histgram).tolist()
            result.append(imagefname.replace("\\", "/"))
            # result.append(imagefname)
            result.append(cat)
            df.loc[len(df)] = result

    print "---------------------"
    print "## write the histograms to file to pass it to the classifier"
    output_path = os.path.abspath("../app/training_data") + "/" + HISTOGRAMS_FILE
    df.to_csv(output_path, index_label="ID")

