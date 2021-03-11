import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from glob import glob


def load_data(file_name):
    print('FILE EXIST')
    featuresDF = pd.read_csv(file_name, sep=';', dtype={'STUDENT': str})
    return featuresDF


def test_classifier(clf_name, clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf_name)
    print(accuracy_score(y_test, y_pred))
    return accuracy_score(y_test, y_pred)


def cross_validate(clf, X, y, features):
    group_kfold = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=5)
    cv_scores = [clf.fit(X[train], y[train]).score(X[test], y[test])
                 for train, test in group_kfold.split(X, y, features['FILE'])]
    return cv_scores


if __name__ == '__main__':

    idx = int(sys.argv[1])
    feature_path = '../mfcc_data'
    # feature_file = glob(feature_path+'/*.csv')[idx]
    feature_file = sorted(glob(feature_path + '/*.csv'))
    feature_file_sorted = sorted(feature_file, key=lambda x: int(x.split('MFCC_')[1].split('.csv')[0]))
    print(feature_file_sorted[idx])

    feature_file = feature_file_sorted[idx]
    features = load_data(feature_file)

    no_mfcc = feature_file.split('\\')[-1].strip('.csv').split('_')[-1]
    results_file = 'resultsMFCC_{}.csv'.format(no_mfcc)
    print(results_file)
    results = pd.DataFrame(columns=['No_MFCC', 'Classifier', 'Accuracy', '5Fold_cv_MEAN', '5Fold_CV', 'Time_sec'])

    # create design matrix X and target vector y
    X = features.filter(like='MFCC').values
    y = features['LABEL_GROUP'].values

    sss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    sss.get_n_splits(X, y, features['FILE'])

    for train_index, test_index in sss.split(X, y, features['FILE']):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    print(pd.DataFrame(y_train)[0].value_counts())

    start_time = time.time()
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    accuracy = test_classifier(type(clf).__name__+str(clf.n_neighbors), clf, X_train, y_train, X_test, y_test)
    training_time = time.time() - start_time
    accuracy_cv = cross_validate(clf, X, y, features)
    results.loc[len(results)] = [no_mfcc, type(clf).__name__+str(clf.n_neighbors), accuracy, np.mean(accuracy_cv), accuracy_cv,  training_time]

    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    accuracy = test_classifier(type(clf).__name__, clf, X_train, y_train, X_test, y_test)
    training_time = time.time() - start_time
    accuracy_cv = cross_validate(clf, X, y, features)
    results.loc[len(results)] = [no_mfcc, type(clf).__name__, accuracy, np.mean(accuracy_cv), accuracy_cv, training_time]

    start_time = time.time()
    clf = ExtraTreesClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    accuracy = test_classifier(type(clf).__name__, clf, X_train, y_train, X_test, y_test)
    training_time = time.time() - start_time
    accuracy_cv = cross_validate(clf, X, y, features)
    results.loc[len(results)] = [no_mfcc, type(clf).__name__, accuracy, np.mean(accuracy_cv), accuracy_cv, training_time]

    start_time = time.time()
    clf = xgb.XGBClassifier(tree_method='approx', random_state=0, n_jobs=-1)
    accuracy = test_classifier(type(clf).__name__, clf, X_train, y_train, X_test, y_test)
    training_time = time.time() - start_time
    accuracy_cv = cross_validate(clf, X, y, features)
    results.loc[len(results)] = [no_mfcc, type(clf).__name__, accuracy, np.mean(accuracy_cv), accuracy_cv, training_time]

    results.to_csv('./results_1_v2/'+results_file, sep=';', float_format='%.4f')
