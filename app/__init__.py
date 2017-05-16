from flask import Flask, request, url_for, render_template
import pandas as pd
import numpy as np
from descriptor.colordescriptor import ColorDescriptor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2
import operator

app = Flask(__name__)
prediction = []

# loading training data
df = pd.read_csv('app/training_data/colorFeatures.csv')

# create design matrix X and target vector y
X = np.array(df.ix[:, 1:289])  # end index is exclusive
y = np.array(df['flower_name'])  # another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# instantiate learning model (k = 5)
knn = KNeighborsClassifier(n_neighbors=10,weights='distance')
# fitting the model
knn.fit(X, y)

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))


def generateHTML(predict_result):
    prediction = []
    for i in predict_result[0]:
        # get file_name
        row = df.ix[i]
        file_name = url_for('static', filename=row['file_name'], _external=True)
        flower_name = row['flower_name']
        dict = {}
        dict['file_name'] = file_name
        dict['flower_name'] = flower_name
        prediction.append(dict)
    return prediction


def generateFinalOutput(predictIDs, sorted_dict):
    result = []
    for tup in sorted_dict:
        dict = {}
        dict["probability"] = tup[1]
        dict["flower_name"] = tup[0]
        image_array = []
        for i in predictIDs:
            row = df.ix[i]
            flower_name = row['flower_name']
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
    else:
        print "SIFT feature"
    color_feature = extractColorFeature("./app/static/upload_img/" + input_image)
    input_feature = np.array(color_feature).reshape(1, -1)
    # get k neighbor
    final = knn.kneighbors(input_feature, return_distance=False)
    final = np.squeeze(final).tolist()

    p = np.squeeze(knn.predict_proba(input_feature)).tolist()
    print p
    flower_name = knn.classes_

    pro_dict = {}
    for i in range(0, len(p)):
        if p[i] > 0:
            pro_dict[flower_name[i]] = p[i]

    print pro_dict
    sorted_dict = sorted(pro_dict.items(), key=operator.itemgetter(1), reverse=True)
    final_output = generateFinalOutput(final, sorted_dict)
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
