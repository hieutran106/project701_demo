import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


CODEBOOK_FILE = 'codebook.file'
HISTOGRAMS_FILE = 'trainingdata_svm.csv'
COLOR_FEATURE_FILE = 'colorFeatures.csv'

if __name__ == '__main__':

    #build classification for color feature


    # loading training data
    df = pd.read_csv(HISTOGRAMS_FILE)
    df.head()

    # create design matrix X and target vector y
    X = np.array(df.ix[:, 1:101]) 	# end index is exclusive
    print X
    y = np.array(df['flower_name']) 	# another way of indexing a pandas df

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(len(X_train)+len(X_test))
    print(len(y_train))

    # instantiate learning model
    svm_model=svm.SVC(kernel='poly')
    print(svm_model)

    # fitting the model
    svm_model.fit(X_train, y_train)
    # predict the response
    pred = svm_model.predict(X_test)

    print(len(pred))
    # evaluate accuracy
    print("The accuracy of SVM algorithm: "+str(accuracy_score(y_test, pred)))

    # instantiate learning model (k = 5)
    knn = KNeighborsClassifier(n_neighbors=11)

    # fitting the model
    knn.fit(X_train, y_train)
    # predict the response
    pred = knn.predict(X_test)

    print(len(pred))
    # evaluate accuracy
    print("The accuracy of kNN algorithm: " + str(accuracy_score(y_test, pred)))