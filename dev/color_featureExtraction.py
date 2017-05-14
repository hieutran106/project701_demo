# import the necessary packages
from descriptor.colordescriptor import ColorDescriptor
import cv2
import os
from os.path import exists, isdir, basename, join, splitext
import numpy
from glob import glob
import pandas as pd


datasetpath = 'dataset_img'
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
OUTPUT_FILE = 'colorFeatures.csv'


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

def extractColorFeature(input_file):
    image = cv2.imread(input_file)
    feature = cd.describe(image)
    return feature


if __name__ == '__main__':
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
    # initialize the color descriptor
    cd = ColorDescriptor((8, 12, 3))
    #init data frame
    columns = range(0, 288)
    columns.append('file_name')
    columns.append('flower_name')
    df = pd.DataFrame(columns=columns)

    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_dic[cat] = cat_files
        for input_file in cat_files:
            feature = extractColorFeature(input_file)
            feature.append(input_file.replace("\\","/"))
            feature.append(cat)
            df.loc[len(df)] = feature

    print "---------------------"
    print "## write color feature into file to pass it to the classifier"
    output_path = os.path.abspath("../app/training_data")+"/"+OUTPUT_FILE
    df.to_csv(output_path, index_label="ID")







