import cPickle

datasetpath = './dataset_img'
PRE_ALLOCATION_BUFFER = 100  # for sift
NUMBER_OF_DESCRIPTOR = 20  # for sift
EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
CODEBOOK_FILE = 'codebook.file'
HISTOGRAMS_FILE = 'trainingdata.svm'




if __name__ == '__main__':
    print "----------------------"
    print "Read codebook from file"
    with open(datasetpath + CODEBOOK_FILE, 'rb') as f:
        codebook = cPickle.load(f)


