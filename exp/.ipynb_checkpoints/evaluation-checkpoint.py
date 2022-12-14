import numpy as np
from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist, pdist
from numpy.random import default_rng
from collections import Counter

def knn_clf(nbr_vec, y):
    '''
    Helper function to generate knn classification result.
    '''
    y_vec = y[nbr_vec]
    c = Counter(y_vec)
    return c.most_common(1)[0][0]

def knn_eval_series(X, y, n_neighbors_list=[1, 2, 3, 4, 5, 10, 15, 20], n_jobs=-1):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An accuracy is calculated by an k-nearest neighbor classifier.
    A series of accuracy will be calculated for the given n_neighbors.
    Input:
        X: A numpy array with the shape [N, k]. The lower dimension embedding
           of some dataset. Expected to have some clusters.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset.
        n_neighbors_list: A list of int.
        kwargs: Any keyword argument that is send into the knn clf.
    Output:
        accs: The avg accuracy generated by the clf, using leave one out cross val.
    '''
    avg_accs = []
    max_acc = X.shape[0]
    # Train once, reuse multiple times
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_list[-1]+1, n_jobs=n_jobs).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices [:, 1:]
    distances = distances[:, 1:]
    for n_neighbors in n_neighbors_list:
        sum_acc = 0
        for i in range(X.shape[0]):
            indices_temp = indices[:, :n_neighbors]
            result = knn_clf(indices_temp[i], y)
            if result == y[i]:
                sum_acc += 1
        avg_acc = sum_acc / max_acc
        avg_accs.append(avg_acc)
    return 1-np.array(avg_accs)

def svm_eval(Z_train, Y_train, n_splits=5, n_jobs=-1):
    skf = StratifiedKFold(n_splits=n_splits)

    sum_acc = 0
    for train_index, test_index in skf.split(Z_train, Y_train):
        clf = svm.LinearSVC(tol=1e-5)
        feature_map_nystroem = Nystroem(n_jobs=n_jobs)
        Z_train_transf = feature_map_nystroem.fit_transform(Z_train[train_index])
        Z_test_transf = feature_map_nystroem.transform(Z_train[test_index])
        clf.fit(Z_train_transf, Y_train[train_index])
        sum_acc += accuracy_score(Y_train[test_index], clf.predict(Z_test_transf))
    
    return sum_acc/n_splits

def random_triplet_eval(X, X_new, y):
    '''
    This is a function that is used to evaluate the lower dimension embedding.
    An triplet satisfaction score is calculated by evaluating how many randomly
    selected triplets have been violated. Each point will generate 5 triplets.
    Input:
        X: A numpy array with the shape [N, p]. The higher dimension embedding
           of some dataset. Expected to have some clusters.
        X_new: A numpy array with the shape [N, k]. The lower dimension embedding
               of some dataset. Expected to have some clusters as well.
        y: A numpy array with the shape [N, 1]. The labels of the original
           dataset. Used to identify clusters
    Output:
        acc: The score generated by the algorithm.
    '''    

    # Sampling Triplets
    # Five triplet per point
    anchors = np.arange(X.shape[0])
    rng = default_rng()
    triplets = rng.choice(anchors, (X.shape[0], 5, 2))
    triplet_labels = np.zeros((X.shape[0], 5))
    anchors = anchors.reshape((-1, 1, 1))
    
    # Calculate the distances and generate labels
    b = np.broadcast(anchors, triplets)
    distances = np.empty(b.shape)
    distances.flat = [np.linalg.norm(X[u] - X[v]) for (u,v) in b]
    labels = distances[:, :, 0] < distances[: , :, 1]

    # Calculate distances for LD
    b = np.broadcast(anchors, triplets)
    distances_l = np.empty(b.shape)
    distances_l.flat = [np.linalg.norm(X_new[u] - X_new[v]) for (u,v) in b]
    pred_vals = distances_l[:, :, 0] < distances_l[:, :, 1]
    correct = np.sum(pred_vals == labels)
    acc = correct/X.shape[0]/5
    return acc

def lof_eval(X, Z, n_jobs=-1):
    clf = LocalOutlierFactor(n_jobs=n_jobs)
    clf.fit(X)
    outlier_factor_input_space = clf.negative_outlier_factor_
    clf = LocalOutlierFactor(n_jobs=n_jobs)
    clf.fit(Z)
    outlier_factor_latent_space = clf.negative_outlier_factor_
    lof_score = np.mean((outlier_factor_input_space-outlier_factor_latent_space)**2)
    return lof_score
 
def isf_eval(X, Z, n_jobs=-1):
    clf = IsolationForest(n_jobs=n_jobs)
    clf.fit(X)
    outlier_factor_input_space = clf.score_samples(X)
    clf = IsolationForest(n_jobs=n_jobs)
    clf.fit(Z)
    outlier_factor_latent_space = clf.score_samples(Z)
    isf_score = np.mean((outlier_factor_input_space-outlier_factor_latent_space)**2)
    return isf_score

def corr_eval(X,Z):
    L = pdist(Z)
    R = pdist(X)
    return np.corrcoef(L.ravel(),R.ravel())[0,1]


def compute_metrics(X, Z, Y, n_jobs=-1):    
    svm_score = svm_eval(Z, Y, n_jobs=n_jobs)
    knn_score = knn_eval_series(Z, Y, n_jobs=n_jobs)
    triplet_score = random_triplet_eval(X, Z, Y)
    lof_score = lof_eval(X, Z, n_jobs=n_jobs)
    isf_score = isf_eval(X, Z, n_jobs=n_jobs)
    corr_score = corr_eval(X, Z)
    return {'SVM':svm_score,
            'KNN':knn_score,
            'Triplet':triplet_score,
            'LOF':lof_score,
            'IsF':isf_score,
            'Corr':corr_score}
