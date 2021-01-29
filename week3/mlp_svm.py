import os
import getpass

from utils import *

from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.misc import imresize
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle as cPickle

#user defined variables
IMG_SIZE = 32

DATASET_DIR = '/home/mcv/datasets/MIT_split'
MODEL_FNAME = '/home/group04/week3/models/multiple_layers/mlp_basic.h5'
RESULTS = '/home/group04/week3/results/svm/'

train_images_filenames = cPickle.load(open('/home/group04/data/train_images_filenames.dat','rb'))
test_images_filenames = cPickle.load(open('/home/group04/data/test_images_filenames.dat','rb'))
train_labels = cPickle.load(open('/home/group04/data/train_labels.dat','rb'))
test_labels = cPickle.load(open('/home/group04/data/test_labels.dat','rb'))

model = load_model(MODEL_FNAME)
layer_names=[layer.name for layer in model.layers]
model_layer = Model(inputs=model.input, outputs=model.get_layer(layer_names[-4]).output)

model_layer.summary()

train_features = []

for i in range(len(train_images_filenames)):
    filename = train_images_filenames[i]
    x = np.asarray(Image.open(filename))
    x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
    x_features = model_layer.predict(x/255.0)
    train_features.append(np.asarray(x_features).reshape(-1))

train_features = np.asarray(train_features)
test_features = []

for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    x = np.asarray(Image.open(filename))
    x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
    x_features = model_layer.predict(x/255.0)
    test_features.append(np.asarray(x_features).reshape(-1))

test_features = np.asarray(test_features)

scaler = StandardScaler()
scaler.fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

parameters = {'kernel': ('rbf', 'linear', 'sigmoid')}
grid = GridSearchCV(svm.SVC(), parameters, n_jobs=3, cv=8)
grid.fit(train_features, train_labels)

best_kernel = grid.best_params_['kernel']

classifier = svm.SVC(kernel=best_kernel)
classifier.fit(train_features,train_labels)

accuracy = classifier.score(test_features, test_labels)

compute_roc(train_features, test_features,train_labels,test_labels, classifier,RESULTS+'ROC_svm.png')

print('Test accuracy: ', accuracy)
