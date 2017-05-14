import os
from os.path import exists, isdir, basename, join, splitext
from glob import glob

datasetpath = './dataset_img'
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]



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



if __name__ == '__main__':
    OUTPUT_FILE="hieu.csv"
    basepath = os.path.dirname(__file__)
    output_path = os.path.join(basepath, "../../app/training_data"+OUTPUT_FILE)
    print(basepath)
    print(output_path)

    