#!/usr/bin/python

import numpy as np
import itertools

from randomforest import weakLearner
from randomforest.weakLearner import FeatureExtractor

from sklearn import ensemble

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import time
import random as rand

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mat_file_name = "coursework1/face.mat"

mat_content = sio.loadmat(mat_file_name)

#mat_content # Let's see the content...  # X: face images, l: label for face
face_data = mat_content['X']

#print(face_data) # Each column represents one face image, each row a pixel value for a particular coordinate of the image
#print(face_data.shape)

face_label = mat_content['l']

#print(face_label)
#print(face_label.shape)
'''Split data into 8:2'''

X_train, X_test, Y_train, Y_test = train_test_split(face_data.T, face_label.T, test_size=0.2, random_state=42, stratify=face_label.T )#random seed for reproducible split
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)
'''compute the average face vector'''
avg_face_numpy = np.mean(X_train, axis = 0)
#plt.imshow(np.reshape(avg_face_numpy,(46,56)).T, cmap = 'gist_gray')


from sklearn.metrics import accuracy_score


for learner in weakLearner.__all__:
    test_class = getattr(weakLearner, learner)()
    params = {'max_depth': None,
    'min_samples_split': 2,
    'n_jobs': 1,
    'n_estimators': 100,
    'oob_score':True, 
    'random_state':123456}

    print(str(learner))

    forest = RandomForestClassifier(**params)
    feature_extractor = FeatureExtractor(test_class, n_features=2576)
    features = feature_extractor.fit_transform(X_train)
    print(features.shape)
    forest.fit(features, Y_train[:,0])

    test_features = feature_extractor.apply_all(X_test)
    print(forest.score(test_features,Y_test[:,0]))


    z = feature_extractor.apply_all(X_test)
    predicted = forest.predict(z)
    accuracy = accuracy_score(Y_test, predicted)

    print("Train set accuracy: {:.3f}".format(forest.score(feature_extractor.apply_all(X_train), Y_train)) )
    print(f'Out-of-bag score estimate: {forest.oob_score_:.3}')
    print("Test set accuracy: {:.3f}".format(forest.score(feature_extractor.apply_all(X_test), Y_test)) )
    print(f'Mean accuracy score: {accuracy:.3}')

    ConfusionMatrixDisplay.from_predictions(Y_test, predicted, include_values=False, colorbar=False)
    plt.show()

    print('Accuracy', accuracy_score(Y_test, predicted))