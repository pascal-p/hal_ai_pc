import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import jaccard_similarity_score, f1_score, log_loss
# from sklearn.metrics import jaccard_score


def plot_for_best_k(mean_acc, std_acc, ks=10):
    rx = range(1, ks + 1)
    plt.plot(rx, mean_acc,'g')
    plt.fill_between(rx, mean_acc - 2 * std_acc, mean_acc + 2 * std_acc, alpha=0.10)
    plt.legend(('Accuracy ', '+/- 2 x std'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.show()
    return

def find_best_k(X_train, y_train, X_val, y_val, ks=10):
    mean_acc = np.zeros((ks))
    std_acc = np.zeros((ks))
    for k in range(0, ks):
        # Train Model and Predict (y_hat)
        knn = KNeighborsClassifier(n_neighbors = k + 1).fit(X_train, y_train)
        y_hat = knn.predict(X_val)
        # Record mean acc and std dev acc
        mean_acc[k] = metrics.accuracy_score(y_val, y_hat)
        std_acc[k] = np.std(y_hat == y_val) / np.sqrt(y_hat.shape[0])
        #
    best_k = mean_acc.argmax()+1
    print( "The best accuracy was:", mean_acc.max(), " with k=", best_k)
    return (best_k, mean_acc, std_acc)

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return ((X - mu) / sigma, mu, sigma)

def feature_normalize_mu_sigma(X, mu, sigma):
    # assert (sigma > 0.0), "[FAIL] sigma cannot be 0."
    return (X - mu) / sigma

def cross_val_norm_score(clf, X, y, cv=5, rs=974, scoring="accuracy"):
    """
    k-fold cross validation combining feature normalization
    """
    skfolds = StratifiedKFold(n_splits=cv, random_state=rs)
    scores = []
    for train_ix, val_ix in skfolds.split(X, y):
        # print("Got indexes: ", [train_ix, val_ix])
        clone_clf = clone(clf)
        #
        X_train_f, y_train_f = np.array(X)[train_ix], np.array(y)[train_ix]
        X_train_f, mu, sig = feature_normalize(X_train_f) # normalize X_train_f
        #
        X_val_f, y_val_f   = np.array(X)[val_ix], np.array(y)[val_ix]
        X_val_f = feature_normalize_mu_sigma(X_val_f, mu, sig) # apply same normalization (as X_train_f)
        #
        clone_clf.fit(X_train_f, y_train_f)  # fit model and
        y_hat = clone_clf.predict(X_val_f)  # predict
        r_correct = sum(y_hat == y_val_f) / len(y_hat)
        scores.append(r_correct)
        #
    return scores

def find_best_k_cv(X_train, y_train, ks=10, cv=5):
    mean_acc = np.zeros((ks))
    std_acc = np.zeros((ks))
    for k in range(0, ks):
        # Train Model and Predict (y_hat)
        knn = KNeighborsClassifier(n_neighbors = k + 1)
        scores = cross_val_norm_score(knn, X_train, y_train, cv=5, scoring="accuracy")
        mean_acc[k] = np.mean(scores)
    best_k = mean_acc.argmax()+1
    print("The best accuracy was: {0:0.5f} with k = {1:1d}".format(mean_acc.max(), best_k))
    return (best_k, mean_acc.max)

def prep_test_set(df, mu, sigma):
    """
    Apply to test set all transformation done on training set, using mu, sigma
    as defined on train set
    """
    # 1 - Date conversion
    # df['due_date'] = pd.to_datetime(df['due_date'])
    df['effective_date'] = pd.to_datetime(df['effective_date'])

    # 2 - Feature selection/extraction
    df['dayofweek'] = df['effective_date'].dt.dayofweek
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)

    # 3 - Convert Categorical features to numerical values
    df['Gender'].replace(to_replace=['male','female'], value=[0, 1], inplace=True)

    # 4 - One Hot Encoding
    feature_df = df[['Principal','terms','age','Gender','weekend']]
    feature_df = pd.concat([feature_df, pd.get_dummies(df['education'])], axis=1)
    feature_df.drop(['Master or Above'], axis = 1, inplace=True)  # ?

    # 5 - normalize data given mu, sigma
    X = feature_normalize_mu_sigma(feature_df, mu, sigma)
    y = df['loan_status']
    return X, y

def conv_to_bin(y):
    return np.where(y == 'PAIDOFF', 1, 0)

def scores_fn(y_test, y_hat, with_log_loss=False):
    y_test_bin, y_hat_bin = conv_to_bin(y_test), conv_to_bin(y_hat)
    # j_sc = jaccard_score(y_test_bin, y_hat_bin)
    j_s_sc = jaccard_similarity_score(y_test_bin, y_hat_bin)
    f1_sc = f1_score(y_test_bin, y_hat_bin)
    ll_sc = log_loss(y_test_bin, y_hat_bin) if with_log_loss else 'NA'
    #
    # print("\tClassisifcation report:\n", classification_report(y_test_bin, y_hat_bin))
    return (j_s_sc, f1_sc, ll_sc) # return (j_sc, j_s_sc, f1_sc, ll_sc)

def print_scores(j_s_sc, f1_sc, ll_sc):
    print("\tjaccard_similarity_score: {0:1.5f}".format(j_s_sc))
    # print("\tjaccard_score: {0:1.5f}".format(j_sc))
    print("\tf1_score: {0:1.5f}".format(f1_sc))
    if ll_sc == 'NA':
        print("\tlogloss_score: NA")
    else:
        print("\tlogloss_score: {0:1.5f}".format(l_sc))
    return

def add_data(algo, j_s_sc, f1_sc, ll_sc='NA', data={}):
    if len(data) == 0:
        data = {'Algorithm': [], 'Jaccard': [], 'F1-score': [], 'LogLoss': []}
    data['Algorithm'].append(algo)
    data['Jaccard'].append(j_s_sc)
    data['F1-score'].append(f1_sc)
    data['LogLoss'].append(ll_sc)
    return data
