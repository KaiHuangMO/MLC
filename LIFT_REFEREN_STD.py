from skmultilearn.dataset import load_dataset
import scipy.io
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings("ignore")
# instantiate the classifier
from skmultilearn.problem_transform import ClassifierChain

from sklearn.metrics import f1_score
from skmultilearn.adapt import MLkNN

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import RakelD
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from LIFTO import LIFT_SAMPLES


testDataset = ['emotions','birds','enron','yeast','scene','bibtex','medical','tmc2007_500','mediamill','Corel5k']

for data in testDataset:
    print ('start' + data)
    X, y, _, _ = load_dataset(data, 'train')
    X_testO, y_test, _, _ = load_dataset(data, 'test')

    X = X.toarray()
    y = y.toarray()

    X_testO = X_testO.toarray()
    y_test = y_test.toarray()

    from sklearn.decomposition import PCA
    print("X.shape: " + str((X.shape)))


    print("X.shape: " + str((X.shape)))
    print("y.shape: " + str((y.shape)))
    print("Descriptive stats:")
    import copy

    macroF1 = {}
    microF1 = {}

    rocs = {}
    aps = {}

    for jj in range(0, 5):
        print (jj)
        random.seed(jj) # 设置随机种子

        X_mlsmote, Y_mlsmote = LIFT_SAMPLES(X,y,X_testO,y_test, perc_gen_instances = .25, K=10, r = .1)
        for kk in range(0, 4):
            br_clf = MLkNN()
            if kk == 0:
                br_clf = MLkNN()
            if kk == 1:
                br_clf = BinaryRelevance(
                    classifier=RandomForestClassifier(),
                    require_dense=[False, True]
                )
            if kk == 2:
                br_clf = RakelD(
                    base_classifier=RandomForestClassifier(),
                    base_classifier_require_dense=[True, True]
                )
                # continue
            if kk == 3:
                br_clf = ClassifierChain(
                    classifier=RandomForestClassifier(),
                    require_dense=[False, True]
                )
            X_mlsmote = np.array(X_mlsmote)
            Y_mlsmote = np.array(Y_mlsmote)
            br_clf.fit(X_mlsmote, Y_mlsmote)
            # predict
            y_pred = br_clf.predict(X_testO)  # .toarray()
            # predict
            y_predp = br_clf.predict_proba(X_testO)
            y_predp = y_predp.toarray()
            # @y_pred2 = np.transpose([pred[:, 1] for pred in y_pred])、
            y_t = y_test

            target_names = []
            for k in range(0, len(y_test[0])):
                target_names.append(str(k))
            #print(classification_report(y_test, y_pred, target_names=target_names))


            #print (f1_score(y_test, y_pred, average='macro'))
            if kk not in macroF1:
                macroF1[kk] = [f1_score(y_test, y_pred, average='macro')]
                microF1[kk] = [f1_score(y_test, y_pred, average='micro')]
                rocs[kk] = [roc_auc_score(y_t, y_predp, multi_class='ovr', average='macro')]
                aps[kk] = [average_precision_score(y_t, y_predp,average='macro')]

            else:
                v1 = macroF1[kk]; v1.append(f1_score(y_test, y_pred, average='macro')) ; macroF1[kk] = v1
                v2 = microF1[kk]; v2.append(f1_score(y_test, y_pred, average='micro')) ; microF1[kk] = v2
                v3 = rocs[kk];v3.append(roc_auc_score(y_t, y_predp, multi_class='ovr', average='macro'));rocs[kk] = v3
                v4 = aps[kk];v4.append(average_precision_score(y_t, y_predp,average='macro'));aps[kk] = v4


    for kk in range(0, 4):
        print(kk)
        print('macroF1 std ' + str(np.round(np.mean(macroF1[kk]), 4)) + ' ' + str(np.round(np.std(macroF1[kk]), 4)))
        print('microF1 std ' + str(np.round(np.mean(microF1[kk]), 4)) + ' ' + str(np.round(np.std(microF1[kk]), 4)))
        print ('rocs std '  + str(np.round(np.mean(rocs[kk]), 4)) + ' ' + str(np.round(np.std(rocs[kk]), 4)))
        print ('aps std '  + str(np.round(np.mean(aps[kk]), 4)) + ' ' + str(np.round(np.std(aps[kk]), 4)))