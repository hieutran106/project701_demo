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
import hashlib

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


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == '__main__':
    sys.stdout = Logger("log/SIFT_extraction_log.txt")
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
    all_features = {}
    cat_label = {}
    cat_dic = {}
    total_descriptor = 0
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_dic[cat] = cat_files
        cat_features, nDes = extractSift(cat_files, mask=True)
        print "Features for dataset_img in category \"", cat, "\":", str(nDes)
        total_descriptor += nDes
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label

    print "Total features extracted:", str(total_descriptor)
    print "##Write total feature to file:"
    output_path = os.path.abspath("../app/training_data") + "/" + TOTAL_FEATURES_FILE
    with open(output_path, 'wb') as f:
        # save all_features into a binary file
        cPickle.dump(all_features, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print "MD5:", md5(output_path)
