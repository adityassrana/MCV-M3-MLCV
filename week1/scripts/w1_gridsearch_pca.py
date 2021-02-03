import os
import cv2
import numpy as np
import pickle as cPickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_validate
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

from sklearn.linear_model import LogisticRegressionCV

from skimage.feature import daisy
import matplotlib.pyplot as plt

from utils import *

def set_params(feat_des='sift',dense=False,level=0,k=128,nfeatures=300,hessianThreshold=600,step=10,pca_perc=0.8):
    params = {
        'feat_des': feat_des, # ['sift','surf','daisy']
        'dense': dense, # ['auto', 'dense']
        'spatial_pyramid_level': level,
        'k': k,
        'nfeatures': nfeatures,
        'hessianThreshold': hessianThreshold,
        'step_size': step,
        'pca_perc': pca_perc,
        'dense_kp': None
    }
    return params

def get_descriptors_D(feat_des_options):
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','rb'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','rb'))
    train_labels = cPickle.load(open('train_labels.dat','rb'))
    test_labels = cPickle.load(open('test_labels.dat','rb'))

    feat_des = create_feat_des(feat_des_options)

    Train_descriptors = []
    Train_label_per_descriptor = []

    train_pyramid_descriptors = []
    for filename,labels in zip(train_images_filenames, train_labels):
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        pyramid_des = spatial_pyramid_des(gray, feat_des, feat_des_options)
        train_pyramid_descriptors.append(pyramid_des)
        Train_descriptors.append(pyramid_des[0])
        Train_label_per_descriptor.append(labels)

    D=np.vstack(Train_descriptors)

    test_pyramid_descriptors = []
    for filename in test_images_filenames:
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        test_pyramid_descriptors.append(spatial_pyramid_des(gray, feat_des, feat_des_options))

    return train_pyramid_descriptors, D, test_pyramid_descriptors

def run(train_pyramid_descriptors, D, test_pyramid_descriptors, feat_des_options):

    train_images_filenames = cPickle.load(open('train_images_filenames.dat','rb'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','rb'))
    train_labels = cPickle.load(open('train_labels.dat','rb'))
    test_labels = cPickle.load(open('test_labels.dat','rb'))

    k = feat_des_options['k']
    codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
    codebook.fit(D)

    visual_words_pyramid=np.zeros((len(train_pyramid_descriptors),k*len(train_pyramid_descriptors[0])),dtype=np.float32)
    for i in range(len(train_pyramid_descriptors)):
        visual_words_pyramid[i,:] = spatial_pyramid_histograms(train_pyramid_descriptors[i], codebook, k)

    knn = KNeighborsClassifier(n_neighbors=5,n_jobs=-1,metric='euclidean')
    knn.fit(visual_words_pyramid, train_labels)

    # logreg = LogisticRegression(random_state=0,max_iter=300).fit(visual_words_pyramid, train_labels)
    # scores = cross_validate(logreg, visual_words_pyramid, train_labels,scoring = ['precision_macro', 'recall_macro','f1_macro'], cv=5,return_estimator=True)

    scores = cross_validate(knn, visual_words_pyramid,
                            train_labels,scoring = ['accuracy', 'precision_macro', 'recall_macro','f1_macro'],
                            cv=8, return_estimator=True)
    cross_val_accuracy = scores['test_accuracy'].mean()
    cross_val_precision = scores['test_precision_macro'].mean()
    cross_val_recall = scores['test_recall_macro'].mean()
    cross_val_f1 = scores['test_f1_macro'].mean()
    # print("%0.2f precision with a std dev of %0.2f" % (cross_val_precision, scores['test_precision_macro'].std()))
    # print("%0.2f recall with a std dev of %0.2f" % (cross_val_recall, scores['test_recall_macro'].std()))
    # print("%0.2f F1-score with a std dev of %0.2f" % (cross_val_f1, scores['test_f1_macro'].std()))

    visual_words_test=np.zeros((len(test_images_filenames),visual_words_pyramid.shape[1]),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        visual_words_test[i,:] = spatial_pyramid_histograms(test_pyramid_descriptors[i], codebook, k)

    test_accuracy = 100*knn.score(visual_words_test, test_labels)
    # print("Test accuracy: %0.2f" % (test_accuracy))

    test_prediction = knn.predict(visual_words_test)
    # test_prediction = logreg.predict(visual_words_test)
    test_precision, test_recall, test_fscore, _ = precision_recall_fscore_support(test_labels, test_prediction, average='macro')
    # print("%0.2f precision" % (test_precision))
    # print("%0.2f recall" % (test_recall))
    # print("%0.2f F1-score" % (test_fscore))

    # pca = PCA(n_components=64)
    pca = PCA(n_components=feat_des_options['pca_perc'], svd_solver='full')
    VWpca = pca.fit_transform(visual_words_pyramid)
    knnpca = KNeighborsClassifier(n_neighbors=5,n_jobs=-1,metric='euclidean')
    knnpca.fit(VWpca, train_labels)
    vwtestpca = pca.transform(visual_words_test)
    pca_test_accuracy = 100*knnpca.score(vwtestpca, test_labels)
    # print("PCA Test accuracy: %0.2f" % (pca_test_accuracy))
    scores_pca = cross_validate(knnpca, visual_words_pyramid, train_labels,
                                scoring = ['accuracy', 'precision_macro', 'recall_macro','f1_macro'],
                                cv=8, return_estimator=True)
    cross_val_accuracy_pca = scores_pca['test_accuracy'].mean()
    cross_val_precision_pca = scores_pca['test_precision_macro'].mean()
    cross_val_recall_pca = scores_pca['test_recall_macro'].mean()
    cross_val_f1_pca = scores_pca['test_f1_macro'].mean()

    lda = LinearDiscriminantAnalysis(n_components=7)
    VWlda = lda.fit_transform(visual_words_pyramid,train_labels)
    knnlda = KNeighborsClassifier(n_neighbors=5,n_jobs=-1,metric='euclidean')
    knnlda.fit(VWlda, train_labels)
    vwtestlda = lda.transform(visual_words_test)
    lda_test_accuracy = 100*knnlda.score(vwtestlda, test_labels)
    # print("LDA Test accuracy: %0.2f" % (lda_test_accuracy))

    return [cross_val_accuracy, cross_val_precision, cross_val_recall, cross_val_f1, test_precision,
           test_recall, test_fscore, test_accuracy, pca_test_accuracy, cross_val_accuracy_pca,
           cross_val_precision_pca, cross_val_recall_pca, cross_val_f1_pca, lda_test_accuracy]
