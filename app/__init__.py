from flask import Flask, request, url_for, render_template
import pandas as pd
import numpy as np
from descriptor.colordescriptor import ColorDescriptor
from descriptor.siftdesciptor import *
from sklearn.neighbors import KNeighborsClassifier
import cv2
import operator
import cPickle

app = Flask(__name__)
CODEBOOK_FILE = 'codebook.file'
prediction = []

# loading training data
df_color = pd.read_csv('app/training_data/colorFeatures.csv')

# create design matrix X and target vector y
X = np.array(df_color.ix[:, 1:289])  # end index is exclusive
y = np.array(df_color['flower_name'])  # another way of indexing a pandas df

# instantiate learning model (k = 10)
knn_color = KNeighborsClassifier(n_neighbors=10, weights='distance')
# fitting the model
knn_color.fit(X, y)

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

#loading traing data for SIFT
df_sift = pd.read_csv('app/training_data/siftFeatures.csv')

with open('app/training_data/'+CODEBOOK_FILE, 'rb') as f:
    # save codebook into a binary file
    codebook=cPickle.load(f)


# create design matrix X and target vector y
X = np.array(df_sift.ix[:, 1:101])  # end index is exclusive
y = np.array(df_sift['flower_name'])  # another way of indexing a pandas df
# instantiate learning model (k = 10)
knn_sift = KNeighborsClassifier(n_neighbors=10, weights='distance')
# fitting the model
knn_sift.fit(X, y)

#init wiki dict
wiki_dict={"Buttercup":"https://en.wikipedia.org/wiki/Ranunculus",
           "Colts Foot":"https://en.wikipedia.org/wiki/Tussilago",
           "Daffodil":"https://en.wikipedia.org/wiki/Narcissus_(plant)",
           "Daisy":"https://en.wikipedia.org/wiki/Daisy",
           "Danlelion":"https://en.wikipedia.org/wiki/Taraxacum",
           "Fritillary":"https://en.wikipedia.org/wiki/Fritillaria",
           "Iris":"https://en.wikipedia.org/wiki/Iris_(plant)",
           "Lily Valley":"https://en.wikipedia.org/wiki/Lily_of_the_valley",
           "Pansy":"https://en.wikipedia.org/wiki/Pansy",
           "Sunflower":"https://en.wikipedia.org/wiki/Helianthus",
           "Tigerlily":"https://en.wikipedia.org/wiki/Lilium_lancifolium",
           "Windflower":"https://en.wikipedia.org/wiki/Anemone"
           }



def generateFinalOutput(predictIDs, sorted_dict,feature='color'):
    result = []
    for tup in sorted_dict:
        dict = {}
        dict["probability"] = int(round(tup[1]*100))
        dict["flower_name"] = tup[0]
        print "flower name:",tup[0]
        dict["wiki_url"]=wiki_dict[tup[0]]
        image_array = []
        for i in predictIDs:
            if feature == 'color':
                row = df_color.ix[i]
            else:
                row = df_sift.ix[i]
            flower_name = row['flower_name']
            print "predict ids:", i, "flower_name:",flower_name
            file_name = url_for('static', filename=row['file_name'], _external=True)
            if flower_name == tup[0]:
                image_array.append(file_name)
        dict["files"] = image_array
        result.append(dict)
    print result
    return result


def extractColorFeature(input_file):
    image = cv2.imread(input_file)
    feature = cd.describe(image)
    return feature


@app.route('/api/predict', methods=['GET'])
def get_tasks():
    input_image = request.args.get("file_name")
    feature = request.args.get("feature")
    if (feature == "color"):
        print "Color feature"
        color_feature = extractColorFeature("./app/static/upload_img/" + input_image)
        input_feature = np.array(color_feature).reshape(1, -1)
        # get k neighbor
        final = knn_color.kneighbors(input_feature, return_distance=False)
        p = np.squeeze(knn_color.predict_proba(input_feature)).tolist()
        flower_name = knn_color.classes_
    else:
        print "SIFT feature"
        kp, des=computeSIFT("./app/static/upload_img/" + input_image)
        input_feature=computeHistograms(codebook,des)
        print "input feature",input_feature
        #input_feature = np.array(des).reshape(1, -1)
        #print input_feature
        print "leng input: ",str(len(input_feature))
        # get k neighbor
        final = knn_sift.kneighbors(input_feature, return_distance=False)
        print "final:",final
        p = np.squeeze(knn_sift.predict_proba(input_feature)).tolist()
        flower_name = knn_sift.classes_
        print "flower_name: ",flower_name

    final = np.squeeze(final).tolist()
    print p


    pro_dict = {}
    for i in range(0, len(p)):
        if p[i] > 0:
            pro_dict[flower_name[i]] = p[i]

    print pro_dict
    sorted_dict = sorted(pro_dict.items(), key=operator.itemgetter(1), reverse=True)
    final_output = generateFinalOutput(final, sorted_dict,feature)
    return render_template('list_image.html', final_output=final_output)


# disable cache
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


from app import views
