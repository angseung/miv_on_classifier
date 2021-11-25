##### DRAFT WRITTEN in 2020.03.24
##### KETI SoC Flatform Research Center
##### IM SEUNGWOO

from sklearn.preprocessing import StandardScaler
import numpy as np

np.set_printoptions(precision=6)


class NotHavePredictMethodError(Exception):
    pass


class NotFittedYetError(Exception):
    pass


def miv(clf=None, X=None, zeta=0.1, threshold=0.9):
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    num_feat = X.shape[1]
    X_1 = X.copy()
    X_2 = X.copy()
    IV = np.zeros(np.shape(X))

    for i in range(num_feat):
        X_1[:, i] = X[:, i] * (1 + zeta)
        X_2[:, i] = X[:, i] * (1 - zeta)
        Y_1 = clf.predict(X_1)
        Y_2 = clf.predict(X_2)
        IV[:, i] = Y_1 - Y_2

        ##RESET DATA
        X_1 = X.copy()
        X_2 = X.copy()

    MIV = IV.mean(0)
    sel_idx = np.argsort(abs(MIV))[::-1]
    sum_MIV = np.sum(abs(MIV))
    cum_MIV = 0
    for i in sel_idx:
        tmp_MIV = abs(MIV)[i] / sum_MIV
        cum_MIV = cum_MIV + tmp_MIV
        print(cum_MIV)

        if cum_MIV >= threshold:
            cum_MIV = cum_MIV - tmp_MIV
            print(cum_MIV)
            th_idx = i - 1
            break

    selected = sel_idx[0 : th_idx + 1]
    return (selected, cum_MIV, IV, MIV)


def miv_mul(*clfs, X=None, zeta=0.1, threshold=0.9):
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    num_feat = X.shape[1]
    X_1 = X.copy()
    X_2 = X.copy()
    (row, col) = X.shape
    len_clf = len(clfs)
    IV = np.zeros((row, col, len_clf))

    for num, clf in enumerate(clfs):
        print(num, type(clf))
        for i in range(num_feat):
            X_1[:, i] = X[:, i] * (1 + zeta)
            X_2[:, i] = X[:, i] * (1 - zeta)
            Y_1 = clf.predict(X_1)
            Y_2 = clf.predict(X_2)
            IV[:, i, num] = Y_1 - Y_2

            ##RESET DATA
            X_1 = X.copy()
            X_2 = X.copy()

    weight = [0.25, 0.75]
    IV = np.average(IV, axis=2, weights=weight)
    MIV = IV.mean(0)
    sel_idx = np.argsort(abs(MIV))[::-1]
    sum_MIV = np.sum(abs(MIV))
    cum_MIV = 0

    for i in sel_idx:
        tmp_MIV = abs(MIV)[i] / sum_MIV
        cum_MIV = cum_MIV + tmp_MIV
        print(cum_MIV)

        if cum_MIV >= threshold:
            cum_MIV = cum_MIV - tmp_MIV
            print(cum_MIV)
            th_idx = i - 1
            break

    selected = sel_idx[0 : th_idx + 1]
    return (selected, cum_MIV, IV, MIV)


def miv_clf(clf=None, X=None, zeta=0.1, threshold=0.9, is_clf=True):
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    num_feat = X.shape[1]
    X_1 = X.copy()
    X_2 = X.copy()
    IV = np.zeros(np.shape(X))

    if is_clf:
        for i in range(num_feat):
            X_1[:, i] = X_1[:, i] * (1.0 + zeta)
            X_2[:, i] = X_2[:, i] * (1.0 - zeta)
            # print(X_1[0, i], X_2[0, i], X[0, i])
            coef_ = clf.coef_
            Y_1 = np.dot(X_1, coef_.T)
            Y_2 = np.dot(X_2, coef_.T)
            Y_1 = Y_1.max(1)
            Y_2 = Y_2.max(1)
            # print(Y_1[0], Y_2[0], "\n")
            IV[:, i] = Y_1 - Y_2

            Y_1 = clf.predict(X_1)
            Y_2 = clf.predict(X_2)
            Y = clf.predict(X)

            acc1 = (Y == Y_1).tolist()
            acc2 = (Y == Y_2).tolist()

            score_1 = sum(acc1) / len(acc1)
            score_2 = sum(acc2) / len(acc2)

            print("%d TH Feature(+%.2f) : %.4f" % (i, zeta, score_1))
            print("%d TH Feature(-%.2f) : %.4f" % (i, zeta, score_2))
            ##RESET DATA
            X_1 = X.copy()
            X_2 = X.copy()
    else:
        for i in range(num_feat):
            X_1[:, i] = X[:, i] * (1 + zeta)
            X_2[:, i] = X[:, i] * (1 - zeta)
            Y_1 = clf.predict(X_1)
            Y_2 = clf.predict(X_2)
            IV[:, i] = Y_1 - Y_2

            ##RESET DATA
            X_1 = X.copy()
            X_2 = X.copy()

    MIV = IV.mean(0)
    sel_idx = np.argsort(abs(MIV))[::-1]
    sum_MIV = np.sum(abs(MIV))
    cum_MIV = 0
    th_idx = 0
    explained_MIV = {}

    for i, val in enumerate(sel_idx):
        tmp_MIV = abs(MIV)[val] / sum_MIV
        explained_MIV[val] = tmp_MIV

    for i, val in enumerate(sel_idx):
        tmp_MIV = abs(MIV)[val] / sum_MIV
        cum_MIV = cum_MIV + tmp_MIV
        # print(cum_MIV)

        if cum_MIV >= threshold:
            cum_MIV = cum_MIV - tmp_MIV
            # print(cum_MIV)
            th_idx = i - 1
            break

    explained_pca = {}
    cov_mat = np.cov(X.T)
    [eigen_vals, eigen_vecs] = np.linalg.eig(cov_mat)
    eigen_val = np.sort(eigen_vals)[::-1]
    eigen_idx = np.argsort(eigen_vals)[::-1]

    for i, val in zip(eigen_idx, eigen_vals):
        explained_pca[i] = val

    selected = sel_idx[0 : th_idx + 1]
    return (selected, cum_MIV, IV, MIV, explained_MIV, explained_pca)
