import pandas as pd
import numpy as np
from descriptor.colordescriptor import ColorDescriptor

from sklearn.neighbors import KNeighborsClassifier
import cv2
import operator

def ttt(predictIDs,sorted_dict):
    result=[]
    print  sorted_dict
    for tup in sorted_dict:
        dict={}
        dict["pro"]=tup[1]
        dict["flower_name"]=tup[0]
        image_array=[]
        for i in predictIDs:
            row = df.ix[i]
            flower_name = row['flower_name']
            file_name=row['file_name']
            if flower_name==tup[0]:
                image_array.append(file_name)
        dict["files"]=image_array
        result.append(dict)
    return result

def find_class(predict_result):
    for i in predict_result[0]:
        # get file_name
        row = df.ix[i]
        flower_name = row['flower_name']
        print "id:",i," flower_name:",flower_name

def extractColorFeature(input_file):
    image = cv2.imread(input_file)
    feature = cd.describe(image)
    return feature

prediction = []

# loading training data
df = pd.read_csv('training_data/colorFeatures.csv')

# create design matrix X and target vector y
X = np.array(df.ix[:, 1:289])  # end index is exclusive
y = np.array(df['flower_name'])  # another way of indexing a pandas df



# instantiate learning model (k = 5)
knn = KNeighborsClassifier(n_neighbors=10,weights='distance')
# fitting the model
knn.fit(X, y)

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))



input_image="static/upload_img/Pansy_Flower.jpg"
color_feature = extractColorFeature(input_image)

input_feature = np.array(color_feature).reshape(1 ,-1)
final = knn.kneighbors(input_feature, return_distance=False)
final=np.squeeze(final).tolist()
p = np.squeeze(knn.predict_proba(input_feature)).tolist()
flower_name= knn.classes_
print p


pro_dict={}
for i in range(0,len(p)):
    if p[i]>0:
        pro_dict[flower_name[i]]=p[i]
        print "Probability:",str(p[i]),"flower_name:",flower_name[i]
print pro_dict
sorted_dict = sorted(pro_dict.items(), key=operator.itemgetter(1),reverse=True)



out=ttt(final,sorted_dict)


print out




