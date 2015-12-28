from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics.pairwise import euclidean_distances as euc_dist
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def get_features(df_train, df_test):
    n_dep = len(np.unique(np.concatenate(
        [df_train['Department_'], df_test['Department_']])))
    n_fn = len(np.unique(np.concatenate(
        [df_train['FinelineNumber_'], df_test['FinelineNumber_']])))
    n_upc = len(np.unique(np.concatenate(
        [df_train['Upc_'], df_test['Upc_']])))

    # labels
    y_train = df_train.groupby(['VisitNumber_']).first()['TripType_']
    Y_train = pd.get_dummies(y_train).as_matrix()
    eps = 2**-52

    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True, use_idf=True)
    # tfidf = TfidfTransformer(norm='l2', sublinear_tf=False, use_idf=True)

    n_br_fn = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'FinelineNumber_']).sum().reset_index()
        g['br'] = np.logical_and(
            g['ScanCount_binary'] > 0, g['ScanCount_binary_neg'] > 0)
        n_br_fn.append(
            g.groupby(['VisitNumber_']).sum().reset_index()['br'])

    n_br_upc = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Upc_', 'ScanCount_binary']).sum().reset_index()
        g['br'] = np.logical_and(
            g['ScanCount_binary'] > 0, g['ScanCount_binary_neg'] > 0)
        n_br_upc.append(
            g.groupby(['VisitNumber_']).sum().reset_index()['br'])

    b_bought = []
    n_bought = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_']).sum().reset_index()
        b_bought.append(g['ScanCount_binary'] > 0)
        n_bought.append(g['ScanCount_rect'])

    b_returned = []
    n_returned = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_']).sum().reset_index()
        b_returned.append(g['ScanCount_binary_neg'] > 0)
        n_returned.append(g['ScanCount_rect_neg'])

    # fn raw and tfidf
    fn = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'FinelineNumber_', 'ScanCount_binary']).sum().reset_index()
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 1]
        s = sparse.csr_matrix(
            (g['ScanCount_rect'], (g['VisitNumber_'], g['FinelineNumber_'])),
            shape=(n, n_fn), dtype='float64')
        fn.append(s)

    # upc raw and tfidf
    upc = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Upc_', 'ScanCount_binary']).sum().reset_index()
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 1]
        s = sparse.csr_matrix(
            (g['ScanCount_rect'], (g['VisitNumber_'], g['Upc_'])),
            shape=(n, n_upc), dtype='float64')
        upc.append(s)

    tfidf.fit(fn[0])
    fn_tfidf = []
    for sm in fn:
        fn_tfidf.append(tfidf.transform(sm))

    print('Getting dot product between mean fn and datasets')
    fn_dot = get_dot(fn, y_train)

    print('Getting dot product between mean fn_tfidf and datasets')
    fn_tfidf_dot = get_dot(fn_tfidf, y_train)

    tfidf.fit(upc[0])
    upc_tfidf = []
    for sm in upc:
        upc_tfidf.append(tfidf.transform(sm))

    print('Doing SVD on Fineline ScanCounts...')
    svd = TruncatedSVD(n_components=100)
    svd.fit(sparse.hstack([fn[0], upc[0]]))
    fnupc_red = []
    for sm1, sm2 in zip(fn, upc):
        fnupc_red.append(svd.transform(sparse.hstack([sm1, sm2])))

    print('Doing SVD on Fineline/UPC TFIDF ScanCounts...')
    svd = TruncatedSVD(n_components=1500)
    svd.fit(sparse.hstack([fn_tfidf[0], upc_tfidf[0]]))

    fnupc_tfidf_red = []
    for sm1, sm2 in zip(fn_tfidf, upc_tfidf):
        fnupc_tfidf_red.append(svd.transform(sparse.hstack([sm1, sm2])))

    print('Doing SVD on Fineline TFIDF ScanCounts...\n')
    svd = TruncatedSVD(n_components=100)
    svd.fit(fn_tfidf[0])
    fn_tfidf_red = []
    for sm in fn_tfidf:
        fn_tfidf_red.append(svd.transform(sm))

    fn_r = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'FinelineNumber_', 'ScanCount_binary']).sum().reset_index()
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 0]
        s = sparse.csr_matrix(
            (g['ScanCount_rect_neg'], (g['VisitNumber_'], g['FinelineNumber_'])),
            shape=(n, n_fn), dtype='float64')
        fn_r.append(s)

    tfidf.fit(fn_r[0])
    fn_r_tfidf = []
    for sm in fn_r:
        fn_r_tfidf.append(tfidf.transform(sm))

    print('Getting dot product between mean fn_r and datasets')
    fn_r_dot = get_dot(fn_r, y_train)

    print('Getting dot product between mean fn_r_tfidf and datasets')
    fn_r_tfidf_dot = get_dot(fn_r_tfidf, y_train)

    print('Doing SVD on Fineline Return TFIDF ScanCounts...')
    svd = TruncatedSVD(n_components=50)
    svd.fit(fn_r_tfidf[0])
    fn_r_tfidf_red = []
    for sm in fn_r_tfidf:
        fn_r_tfidf_red.append(svd.transform(sm))

    # #########################################
    print('Doing SVD on Fineline Difference ScanCounts...\n')
    diff_br = []
    diff_br.append(fn[0] - fn_r[0])
    diff_br.append(fn[1] - fn_r[1])
    svd = TruncatedSVD(n_components=100)
    svd.fit(diff_br[0])
    diff_br_red = []
    for sm in diff_br:
        diff_br_red.append(svd.transform(sm))

    # department total scan counts
    dep = []
    dep_p = []
    dep_entropy = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Department_', 'ScanCount_binary']).sum().reset_index()
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 1]
        s = sparse.csr_matrix(
            (g['ScanCount_rect'], (g['VisitNumber_'], g['Department_'])),
            shape=(n, n_dep), dtype='float64')
        dep.append(s.toarray())

        m = s.toarray()
        p = m / np.sum(m, axis=1)[:, np.newaxis]
        p[np.isnan(p)] = 0
        entropy = -np.sum(p * np.log(p + eps), axis=1)
        dep_p.append(p)
        dep_entropy.append(entropy)

    tfidf.fit(dep[0])
    dep_tfidf = []
    for sm in dep:
        dep_tfidf.append(tfidf.transform(sm).toarray())

    sim_matrix = cos_sim(dep[0].T, dep[0].T)
    sim_matrix /= np.sum(sim_matrix, axis=0)
    dep = [d.dot(sim_matrix) for d in dep]

    # dep = [np.log(i + 1) for i in dep]

    print('Getting dot product between mean dep and datasets')
    dep_dot = get_dot(dep, y_train)

    print('Getting dot product between mean dep_p and datasets')
    dep_p_dot = get_dot(dep_p, y_train)

    print('Getting dot product between mean dep_tfidf and datasets')
    dep_tfidf_dot = get_dot(dep_tfidf, y_train)

    print('Getting distances between mean dep and datasets')
    # dep_maha = get_mahalanobis(dep, y_train)
    dep_manh = get_manhattan(dep, y_train)

    print('Getting distances between mean dep_p and datasets')
    # dep_p_maha = get_mahalanobis(dep_p, y_train)
    dep_p_manh = get_manhattan(dep_p, y_train)

    print('Getting distances between mean dep_tfidf and datasets')
    # dep_tfidf_maha = get_mahalanobis(dep_tfidf, y_train)
    dep_tfidf_manh = get_manhattan(dep_tfidf, y_train)

    print('Getting euclidean for dep')
    dep_euclidean = get_euclidean(dep, y_train)

    print('Getting euclidean for dep_p')
    dep_p_euclidean = get_euclidean(dep_p, y_train)

    print('Getting euclidean for dep_tfidf\n')
    dep_tfidf_euclidean = get_euclidean(dep_tfidf, y_train)

    print('Getting cosine for dep')
    dep_cosine = get_cosine(dep, y_train)

    print('Getting cosine for dep_p')
    dep_p_cosine = get_cosine(dep_p, y_train)

    print('Getting cosine for dep_tfidf\n')
    dep_tfidf_cosine = get_cosine(dep_tfidf, y_train)

    enc = OneHotEncoder(n_values=n_dep)
    enc.fit(np.argmax(dep_p[0], axis=1).reshape(-1, 1))
    top_dep = []
    for m in dep_p:
        onehot = enc.transform(np.argmax(m, axis=1).reshape(-1, 1)).toarray()
        no_buy = m.sum(axis=1) == 0
        onehot[no_buy, :] = 0
        top_dep.append(onehot)

    dep_sorted = []
    dep_p_sorted = []
    for m1, m2 in zip(dep, dep_p):
        dep_sorted.append(np.sort(m1, axis=1)[:, -20:])
        dep_p_sorted.append(np.sort(m2, axis=1)[:, -20:])

    dep_sorted = [np.log(i + 1) for i in dep_sorted]

    print('Getting dot product between mean dep_sorted and datasets')
    dep_sorted_dot = get_dot(dep_sorted, y_train)

    print('Getting dot product between mean dep_p_sorted and datasets')
    dep_p_sorted_dot = get_dot(dep_p_sorted, y_train)

    print('Getting distances between mean dep_sorted and datasets')
    # dep_sorted_maha = get_mahalanobis(dep_sorted, y_train)
    dep_sorted_manh = get_manhattan(dep_sorted, y_train)

    print('Getting distances between mean dep_p_sorted and datasets')
    # dep_p_sorted_maha = get_mahalanobis(dep_p_sorted, y_train)
    dep_p_sorted_manh = get_manhattan(dep_p_sorted, y_train)

    print('Getting euclidean for dep_sorted')
    dep_sorted_euclidean = get_euclidean(dep_sorted, y_train)

    print('Getting euclidean for dep_p_sorted\n')
    dep_p_sorted_euclidean = get_euclidean(dep_p_sorted, y_train)

    print('Getting cosine for dep_sorted')
    dep_sorted_cosine = get_cosine(dep_sorted, y_train)

    print('Getting cosine for dep_p_sorted\n')
    dep_p_sorted_cosine = get_cosine(dep_p_sorted, y_train)

    # department unique UPCs
    dep_uniq = []
    dep_uniq_p = []
    dep_uniq_entropy = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Department_', 'ScanCount_binary']).size().reset_index()
        g.rename(columns={0: 'n_unique'}, inplace=True)
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 1]
        s = sparse.csr_matrix(
            (g['n_unique'], (g['VisitNumber_'], g['Department_'])),
            shape=(n, n_dep), dtype='float64')
        dep_uniq.append(s.toarray())

        m = s.toarray()
        p = m / np.sum(m, axis=1)[:, np.newaxis]
        p[np.isnan(p)] = 0
        entropy = -np.sum(p * np.log(p + eps), axis=1)
        dep_uniq_p.append(p)
        dep_uniq_entropy.append(entropy)

    tfidf.fit(dep_uniq[0])
    dep_uniq_tfidf = []
    for sm in dep_uniq:
        dep_uniq_tfidf.append(tfidf.transform(sm).toarray())

    sim_matrix = cos_sim(dep_uniq[0].T, dep_uniq[0].T)
    sim_matrix /= np.sum(sim_matrix, axis=0)
    dep_uniq = [d.dot(sim_matrix) for d in dep_uniq]

    # dep_uniq = [np.log(i + 1) for i in dep_uniq]

    print('Getting dot product between mean dep_uniq and datasets')
    dep_uniq_dot = get_dot(dep_uniq, y_train)

    print('Getting dot product between mean dep_uniq_p and datasets')
    dep_uniq_p_dot = get_dot(dep_uniq_p, y_train)

    print('Getting dot product between mean dep_uniq_tfidf and datasets')
    dep_uniq_tfidf_dot = get_dot(dep_uniq_tfidf, y_train)

    print('Getting distances between mean dep_uniq and datasets')
    # dep_uniq_maha = get_mahalanobis(dep_uniq, y_train)
    dep_uniq_manh = get_manhattan(dep_uniq, y_train)

    print('Getting distances between mean dep_uniq_p and datasets')
    # dep_uniq_p_maha = get_mahalanobis(dep_uniq_p, y_train)
    dep_uniq_p_manh = get_manhattan(dep_uniq_p, y_train)

    print('Getting distances between mean dep_uniq_tfidf and datasets')
    # dep_uniq_tfidf_maha = get_mahalanobis(dep_uniq_tfidf, y_train)
    dep_uniq_tfidf_manh = get_manhattan(dep_uniq_tfidf, y_train)

    print('Getting euclidean for dep_uniq')
    dep_uniq_euclidean = get_euclidean(dep_uniq, y_train)

    print('Getting euclidean dep_uniq_p')
    dep_uniq_p_euclidean = get_euclidean(dep_uniq_p, y_train)

    print('Getting euclidean for mean dep_uniq_tfidf\n')
    dep_uniq_tfidf_euclidean = get_euclidean(dep_uniq_tfidf, y_train)

    print('Getting cosine for dep_uniq')
    dep_uniq_cosine = get_cosine(dep_uniq, y_train)

    print('Getting cosine dep_uniq_p')
    dep_uniq_p_cosine = get_cosine(dep_uniq_p, y_train)

    print('Getting cosine for mean dep_uniq_tfidf\n')
    dep_uniq_tfidf_cosine = get_cosine(dep_uniq_tfidf, y_train)

    # department unique Finelines
    dep_uniq_fn = []
    dep_uniq_fn_p = []
    dep_uniq_fn_entropy = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Department_', 'FinelineNumber_', 'ScanCount_binary']).size().reset_index()
        g.rename(columns={0: 'n_unique'}, inplace=True)
        g['n_unique'][g['n_unique'] > 1] = 1
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 1]
        s = sparse.csr_matrix(
            (g['n_unique'], (g['VisitNumber_'], g['Department_'])),
            shape=(n, n_dep), dtype='float64')
        dep_uniq_fn.append(s.toarray())

        m = s.toarray()
        p = m / np.sum(m, axis=1)[:, np.newaxis]
        p[np.isnan(p)] = 0
        entropy = -np.sum(p * np.log(p + eps), axis=1)
        dep_uniq_fn_p.append(p)
        dep_uniq_fn_entropy.append(entropy)

    tfidf.fit(dep_uniq_fn[0])
    dep_uniq_fn_tfidf = []
    for sm in dep_uniq_fn:
        dep_uniq_fn_tfidf.append(tfidf.transform(sm).toarray())

    sim_matrix = cos_sim(dep_uniq_fn[0].T, dep_uniq_fn[0].T)
    sim_matrix /= np.sum(sim_matrix, axis=0)
    dep_uniq_fn = [d.dot(sim_matrix) for d in dep_uniq_fn]

    # dep_uniq_fn = [np.log(i + 1) for i in dep_uniq_fn]

    print('Getting dot product between mean dep_uniq_fn and datasets')
    dep_uniq_fn_dot = get_dot(dep_uniq_fn, y_train)

    print('Getting dot product between mean dep_uniq_fn_tfidf and datasets')
    dep_uniq_fn_tfidf_dot = get_dot(dep_uniq_fn_tfidf, y_train)

    print('Getting dot product between mean dep_uniq_fn_p and datasets')
    dep_uniq_fn_p_dot = get_dot(dep_uniq_fn_p, y_train)

    print('Getting distances between mean dep_uniq_fn and datasets')
    # dep_uniq_fn_maha = get_mahalanobis(dep_uniq_fn, y_train)
    dep_uniq_fn_manh = get_manhattan(dep_uniq_fn, y_train)

    print('Getting distances between mean dep_uniq_fn_tfidf and datasets')
    # dep_uniq_fn_tfidf_maha = get_mahalanobis(dep_uniq_fn_tfidf, y_train)
    dep_uniq_fn_tfidf_manh = get_manhattan(dep_uniq_fn_tfidf, y_train)

    print('Getting distances between mean dep_uniq_fn_p and datasets')
    # dep_uniq_fn_p_maha = get_mahalanobis(dep_uniq_fn_p, y_train)
    dep_uniq_fn_p_manh = get_manhattan(dep_uniq_fn_p, y_train)

    print('Getting euclidean for dep_uniq_fn')
    dep_uniq_fn_euclidean = get_euclidean(dep_uniq_fn, y_train)

    print('Getting euclidean for dep_uniq_fn_tfidf')
    dep_uniq_fn_tfidf_euclidean = get_euclidean(dep_uniq_fn_tfidf, y_train)

    print('Getting euclidean for mean dep_uniq_fn_p\n')
    dep_uniq_fn_p_euclidean = get_euclidean(dep_uniq_fn_p, y_train)

    print('Getting cosine for dep_uniq_fn')
    dep_uniq_fn_cosine = get_cosine(dep_uniq_fn, y_train)

    print('Getting cosine for dep_uniq_fn_tfidf')
    dep_uniq_fn_tfidf_cosine = get_cosine(dep_uniq_fn_tfidf, y_train)

    print('Getting cosine for mean dep_uniq_fn_p\n')
    dep_uniq_fn_p_cosine = get_cosine(dep_uniq_fn_p, y_train)

    dep_uniq_fn_sorted = []
    dep_uniq_fn_p_sorted = []
    for m1, m2 in zip(dep_uniq_fn, dep_uniq_fn_p):
        dep_uniq_fn_sorted.append(np.sort(m1, axis=1)[:, -20:])
        dep_uniq_fn_p_sorted.append(np.sort(m2, axis=1)[:, -20:])

    dep_uniq_fn_sorted = [np.log(i + 1) for i in dep_uniq_fn_sorted]

    print('Getting dot product between mean dep_uniq_fn_sorted and datasets')
    dep_uniq_fn_sorted_dot = get_dot(dep_uniq_fn_sorted, y_train)

    print('Getting dot product between mean dep_uniq_fn_p_sorted and datasets')
    dep_uniq_fn_p_sorted_dot = get_dot(dep_uniq_fn_p_sorted, y_train)

    print('Getting distances between mean dep_uniq_fn_sorted and datasets')
    # dep_uniq_fn_sorted_maha = get_mahalanobis(dep_uniq_fn_sorted, y_train)
    dep_uniq_fn_sorted_manh = get_manhattan(dep_uniq_fn_sorted, y_train)

    print('Getting distances between mean dep_uniq_fn_p_sorted and datasets')
    dep_uniq_fn_p_sorted_manh = get_manhattan(dep_uniq_fn_p_sorted, y_train)

    print('Getting euclidean for dep_uniq_fn_sorted')
    dep_uniq_fn_sorted_euclidean = get_euclidean(dep_uniq_fn_sorted, y_train)

    print('Getting for dep_uniq_fn_p_sorted\n')
    dep_uniq_fn_p_sorted_euclidean = get_euclidean(dep_uniq_fn_p_sorted, y_train)

    print('Getting cosine for dep_uniq_fn_sorted')
    dep_uniq_fn_sorted_cosine = get_cosine(dep_uniq_fn_sorted, y_train)

    print('Getting for dep_uniq_fn_p_sorted\n')
    dep_uniq_fn_p_sorted_cosine = get_cosine(dep_uniq_fn_p_sorted, y_train)

    # departments scan binaries
    dep_bin = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Department_', 'ScanCount_binary']).sum().reset_index()
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 1]
        s = sparse.csr_matrix(
            (g['ScanCount_binary'], (g['VisitNumber_'], g['Department_'])),
            shape=(n, n_dep), dtype='float64')
        dep_bin.append(s.toarray())

    tfidf.fit(dep_bin[0])
    dep_bin_tfidf = []
    for sm in dep_bin:
        dep_bin_tfidf.append(tfidf.transform(sm).toarray())

    sim_matrix = cos_sim(dep_bin[0].T, dep_bin[0].T)
    sim_matrix /= np.sum(sim_matrix, axis=0)
    dep_bin = [d.dot(sim_matrix) for d in dep_bin]

    print('Getting dot product between mean dep_bin and datasets')
    dep_bin_dot = get_dot(dep_bin, y_train)

    print('Getting dot product between mean dep_bin_tfidf and datasets')
    dep_bin_tfidf_dot = get_dot(dep_bin_tfidf, y_train)

    print('Getting distances between mean dep_bin and datasets')
    # dep_bin_maha = get_mahalanobis(dep_bin, y_train)
    dep_bin_manh = get_manhattan(dep_bin, y_train)

    print('Getting distances between mean dep_bin_tfidf and datasets')
    # dep_bin_tfidf_maha = get_mahalanobis(dep_bin_tfidf, y_train)
    dep_bin_tfidf_manh = get_manhattan(dep_bin_tfidf, y_train)

    print('Getting euclidean for dep_bin')
    dep_bin_euclidean = get_euclidean(dep_bin, y_train)

    print('Getting euclidean for dep_bin_tfidf\n')
    dep_bin_tfidf_euclidean = get_euclidean(dep_bin_tfidf, y_train)

    print('Getting cosine for dep_bin')
    dep_bin_cosine = get_cosine(dep_bin, y_train)

    print('Getting cosine for dep_bin_tfidf\n')
    dep_bin_tfidf_cosine = get_cosine(dep_bin_tfidf, y_train)

    # departments returns
    dep_r = []
    dep_r_p = []
    dep_r_entropy = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Department_', 'ScanCount_binary']).sum().reset_index()
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 0]
        s = sparse.csr_matrix(
            (g['ScanCount_rect_neg'], (g['VisitNumber_'], g['Department_'])),
            shape=(n, n_dep), dtype='float64')
        dep_r.append(s.toarray())

        m = s.toarray()
        p = m / np.sum(m, axis=1)[:, np.newaxis]
        p[np.isnan(p)] = 0
        entropy = -np.sum(p * np.log(p + eps), axis=1)
        dep_r_p.append(p)
        dep_r_entropy.append(entropy)

    tfidf.fit(dep_r[0])
    dep_r_tfidf = []
    for sm in dep_r:
        dep_r_tfidf.append(tfidf.transform(sm).toarray())

    sim_matrix = cos_sim(dep_r[0].T, dep_r[0].T)
    sim_matrix[np.diag_indices(sim_matrix.shape[0])] = 1
    sim_matrix /= np.sum(sim_matrix, axis=0)
    dep_r = [d.dot(sim_matrix) for d in dep_r]

    # dep_r = [np.log(i + 1) for i in dep_r]

    print('Getting dot product between mean dep_r and datasets')
    dep_r_dot = get_dot(dep_r, y_train)

    print('Getting dot product between mean dep_r_tfidf and datasets')
    dep_r_tfidf_dot = get_dot(dep_r_tfidf, y_train)

    print('Getting distances between mean dep_r and datasets')
    dep_r_manh = get_manhattan(dep_r, y_train)

    print('Getting distances between mean dep_r_tfidf and datasets')
    dep_r_tfidf_manh = get_manhattan(dep_r_tfidf, y_train)

    print('Getting euclidean for dep_r')
    dep_r_euclidean = get_euclidean(dep_r, y_train)

    print('Getting euclidean for dep_r_tfidf\n')
    dep_r_tfidf_euclidean = get_euclidean(dep_r_tfidf, y_train)

    print('Getting cosine for dep_r')
    dep_r_cosine = get_cosine(dep_r, y_train)

    print('Getting cosine for dep_r_tfidf\n')
    dep_r_tfidf_cosine = get_cosine(dep_r_tfidf, y_train)

    dep_bought_mr = []
    dep_r_sorted = []
    dep_r_p_sorted = []
    for i, (m1, m2) in enumerate(zip(dep_r, dep_r_p)):
        n = dep[i].shape[0]
        no_buy = dep_p[i].sum(axis=1) == 0

        temp = dep[i][np.arange(n), np.argmax(m1, axis=1)]
        temp[no_buy] = 0
        dep_bought_mr.append(temp)

        dep_r_sorted.append(np.sort(m1, axis=1)[:, -5:])
        dep_r_p_sorted.append(np.sort(m2, axis=1)[:, -5:])

    # departments uniques return
    dep_r_uniq = []
    dep_r_uniq_p = []
    dep_r_uniq_entropy = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Department_', 'ScanCount_binary']).size().reset_index()
        g.rename(columns={0: 'n_unique'}, inplace=True)
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 0]
        s = sparse.csr_matrix(
            (g['n_unique'], (g['VisitNumber_'], g['Department_'])),
            shape=(n, n_dep), dtype='float64')
        dep_r_uniq.append(s.toarray())

        m = s.toarray()
        p = m / np.sum(m, axis=1)[:, np.newaxis]
        p[np.isnan(p)] = 0
        entropy = -np.sum(p * np.log(p + eps), axis=1)
        dep_r_uniq_p.append(p)
        dep_r_uniq_entropy.append(entropy)

    tfidf.fit(dep_r_uniq[0])
    dep_r_uniq_tfidf = []
    for sm in dep_r_uniq:
        dep_r_uniq_tfidf.append(tfidf.transform(sm).toarray())

    sim_matrix = cos_sim(dep_r_uniq[0].T, dep_r_uniq[0].T)
    sim_matrix[np.diag_indices(sim_matrix.shape[0])] = 1
    sim_matrix /= np.sum(sim_matrix, axis=0)
    dep_r_uniq = [d.dot(sim_matrix) for d in dep_r_uniq]

    # dep_r_uniq = [np.log(i + 1) for i in dep_r_uniq]

    # departments scan binaries returned
    dep_r_bin = []
    for df in [df_train, df_test]:
        g = df.groupby(
            ['VisitNumber_', 'Department_', 'ScanCount_binary']).sum().reset_index()
        n = len(np.unique(df['VisitNumber_']))
        g = g[g['ScanCount_binary'] == 0]
        s = sparse.csr_matrix(
            (g['ScanCount_binary_neg'], (g['VisitNumber_'], g['Department_'])),
            shape=(n, n_dep), dtype='float64')
        dep_r_bin.append(s.toarray())

    tfidf.fit(dep_r_bin[0])
    dep_r_bin_tfidf = []
    for sm in dep_r_bin:
        dep_r_bin_tfidf.append(tfidf.transform(sm).toarray())

    sim_matrix = cos_sim(dep_r_bin[0].T, dep_r_bin[0].T)
    sim_matrix[np.diag_indices(sim_matrix.shape[0])] = 1
    sim_matrix /= np.sum(sim_matrix, axis=0)
    dep_r_bin = [d.dot(sim_matrix) for d in dep_r_bin]

    print('Getting dot product between mean dep_r_bin and datasets')
    dep_r_bin_dot = get_dot(dep_r_bin, y_train)

    print('Getting dot product between mean dep_r_bin_tfidf and datasets')
    dep_r_bin_tfidf_dot = get_dot(dep_r_bin_tfidf, y_train)

    print('Getting distances between mean dep_r_bin and datasets')
    dep_r_bin_manh = get_manhattan(dep_r_bin, y_train)

    print('Getting distances between mean dep_r_bin_tfidf and datasets')
    dep_r_bin_tfidf_manh = get_manhattan(dep_r_bin_tfidf, y_train)

    print('Getting euclidean for dep_r_bin\n')
    dep_r_bin_euclidean = get_euclidean(dep_r_bin, y_train)

    print('Getting euclidean for dep_r_bin_tfidf\n')
    dep_r_bin_tfidf_euclidean = get_euclidean(dep_r_bin_tfidf, y_train)

    print('Getting cosine for dep_r_bin\n')
    dep_r_bin_cosine = get_cosine(dep_r_bin, y_train)

    print('Getting cosine for dep_r_bin_tfidf\n')
    dep_r_bin_tfidf_cosine = get_cosine(dep_r_bin_tfidf, y_train)

    n_unique_dep = []
    for df in [df_train, df_test]:
        n_unique_dep.append(
            df.groupby(['VisitNumber_', 'Department_']).size().reset_index().
            groupby(['VisitNumber_']).size().as_matrix())

    n_unique_fn = []
    for df in [df_train, df_test]:
        n_unique_fn.append(
            df.groupby(['VisitNumber_', 'FinelineNumber_']).size().reset_index().
            groupby(['VisitNumber_']).size().as_matrix())

    n_unique_upc = []
    for df in [df_train, df_test]:
        n_unique_upc.append(
            df.groupby(['VisitNumber_', 'Upc_']).size().reset_index().
            groupby(['VisitNumber_']).size().as_matrix())

    max_scan_count = []
    for df in [df_train, df_test]:
        max_scan_count.append(df.groupby(['VisitNumber_'])['ScanCount'].max())

    min_scan_count = []
    for df in [df_train, df_test]:
        min_scan_count.append(df.groupby(['VisitNumber_'])['ScanCount'].min())

    mean_scan_count_per_dep = []
    for i, df in enumerate([df_train, df_test]):
        mean_scan_count_per_dep.append(
            1. * df.groupby(['VisitNumber_'])['ScanCount'].sum() / n_unique_dep[i])

    # Weekday
    onehot = OneHotEncoder()
    day_train = onehot.fit_transform(
        df_train.groupby(['VisitNumber_'])
        .first()['Weekday_'][:, np.newaxis]).toarray()
    day_test = onehot.fit_transform(
        df_test.groupby(['VisitNumber_'])
        .first()['Weekday_'][:, np.newaxis]).toarray()

    X_train = np.c_[
        fnupc_red[0],
        fnupc_tfidf_red[0],
        # fn_tfidf_red[0],
        fn_r_tfidf_red[0],
        diff_br_red[0],

        dep[0], dep_tfidf[0], dep_p[0], dep_entropy[0],
        dep_uniq[0], dep_uniq_tfidf[0], dep_uniq_p[0], dep_uniq_entropy[0],
        dep_uniq_fn[0], dep_uniq_fn_tfidf[0], dep_uniq_fn_p[0], dep_uniq_fn_entropy[0],
        dep_bin[0], dep_bin_tfidf[0],

        dep_sorted[0], dep_p_sorted[0], dep_uniq_fn_sorted[0], dep_uniq_fn_p_sorted[0],

        dep_r[0], dep_r_tfidf[0],
        # dep_r_uniq_tfidf[0],
        dep_r_bin[0], dep_r_bin_tfidf[0],

        dep_r_sorted[0], dep_r_p_sorted[0],

        dep_bought_mr[0],

        top_dep[0],
        n_br_fn[0], # n_br_upc[0],
        b_bought[0], n_bought[0],
        b_returned[0], n_returned[0],
        n_unique_dep[0], n_unique_fn[0], n_unique_upc[0],
        max_scan_count[0], min_scan_count[0], mean_scan_count_per_dep[0],
        day_train,

        fn_dot[0], fn_tfidf_dot[0],
        dep_dot[0], dep_tfidf_dot[0], dep_p_dot[0],
        dep_uniq_dot[0], dep_uniq_tfidf_dot[0], dep_uniq_p_dot[0],
        dep_uniq_fn_dot[0], dep_uniq_fn_tfidf_dot[0], dep_uniq_fn_p_dot[0],
        dep_bin_dot[0], dep_bin_tfidf_dot[0],
        dep_sorted_dot[0], dep_p_sorted_dot[0],
        dep_uniq_fn_sorted_dot[0], dep_uniq_fn_p_sorted_dot[0],

        dep_manh[0], dep_tfidf_manh[0], dep_p_manh[0],
        dep_uniq_manh[0], dep_uniq_tfidf_manh[0], dep_uniq_p_manh[0],
        dep_uniq_fn_manh[0], dep_uniq_fn_tfidf_manh[0], dep_uniq_fn_p_manh[0],
        dep_bin_manh[0], dep_bin_tfidf_manh[0],
        dep_sorted_manh[0], dep_p_sorted_manh[0],
        dep_uniq_fn_sorted_manh[0], dep_uniq_fn_p_sorted_manh[0],

        dep_euclidean[0], dep_tfidf_euclidean[0], dep_p_euclidean[0],
        dep_uniq_euclidean[0], dep_uniq_tfidf_euclidean[0], dep_uniq_p_euclidean[0],
        dep_uniq_fn_euclidean[0], dep_uniq_fn_tfidf_euclidean[0], dep_uniq_fn_p_euclidean[0],
        dep_bin_euclidean[0], dep_bin_tfidf_euclidean[0],
        dep_sorted_euclidean[0], dep_p_sorted_euclidean[0],
        dep_uniq_fn_sorted_euclidean[0], dep_uniq_fn_p_sorted_euclidean[0],

        dep_cosine[0], dep_tfidf_cosine[0], dep_p_cosine[0],
        dep_uniq_cosine[0], dep_uniq_tfidf_cosine[0], dep_uniq_p_cosine[0],
        dep_uniq_fn_cosine[0], dep_uniq_fn_tfidf_cosine[0], dep_uniq_fn_p_cosine[0],
        dep_bin_cosine[0], dep_bin_tfidf_cosine[0],
        dep_sorted_cosine[0], dep_p_sorted_cosine[0],
        dep_uniq_fn_sorted_cosine[0], dep_uniq_fn_p_sorted_cosine[0],

        fn_r_dot[0], fn_r_tfidf_dot[0],

        dep_r_dot[0], dep_r_bin_dot[0],
        dep_r_tfidf_dot[0], dep_r_bin_tfidf_dot[0],

        dep_r_manh[0], dep_r_bin_manh[0],
        dep_r_tfidf_manh[0], dep_r_bin_tfidf_manh[0],

        dep_r_euclidean[0], dep_r_bin_euclidean[0],
        dep_r_tfidf_euclidean[0], dep_r_bin_tfidf_euclidean[0],

        dep_r_cosine[0], dep_r_bin_cosine[0],
        dep_r_tfidf_cosine[0], dep_r_bin_tfidf_cosine[0],
    ]

    X_test = np.c_[
        fnupc_red[1],
        fnupc_tfidf_red[1],
        # fn_tfidf_red[1],
        fn_r_tfidf_red[1],
        diff_br_red[1],

        dep[1], dep_tfidf[1], dep_p[1], dep_entropy[1],
        dep_uniq[1], dep_uniq_tfidf[1], dep_uniq_p[1], dep_uniq_entropy[1],
        dep_uniq_fn[1], dep_uniq_fn_tfidf[1], dep_uniq_fn_p[1], dep_uniq_fn_entropy[1],
        dep_bin[1], dep_bin_tfidf[1],

        dep_sorted[1], dep_p_sorted[1], dep_uniq_fn_sorted[1], dep_uniq_fn_p_sorted[1],

        dep_r[1], dep_r_tfidf[1],
        # dep_r_uniq_tfidf[1],
        dep_r_bin[1], dep_r_bin_tfidf[1],

        dep_r_sorted[1], dep_r_p_sorted[1],

        dep_bought_mr[1],

        top_dep[1],
        n_br_fn[1], # n_br_upc[1],
        b_bought[1], n_bought[1],
        b_returned[1], n_returned[1],
        n_unique_dep[1], n_unique_fn[1], n_unique_upc[1],
        max_scan_count[1], min_scan_count[1], mean_scan_count_per_dep[1],
        day_test,

        fn_dot[1], fn_tfidf_dot[1],
        dep_dot[1], dep_tfidf_dot[1], dep_p_dot[1],
        dep_uniq_dot[1], dep_uniq_tfidf_dot[1], dep_uniq_p_dot[1],
        dep_uniq_fn_dot[1], dep_uniq_fn_tfidf_dot[1], dep_uniq_fn_p_dot[1],
        dep_bin_dot[1], dep_bin_tfidf_dot[1],
        dep_sorted_dot[1], dep_p_sorted_dot[1],
        dep_uniq_fn_sorted_dot[1], dep_uniq_fn_p_sorted_dot[1],

        dep_manh[1], dep_tfidf_manh[1], dep_p_manh[1],
        dep_uniq_manh[1], dep_uniq_tfidf_manh[1], dep_uniq_p_manh[1],
        dep_uniq_fn_manh[1], dep_uniq_fn_tfidf_manh[1], dep_uniq_fn_p_manh[1],
        dep_bin_manh[1], dep_bin_tfidf_manh[1],
        dep_sorted_manh[1], dep_p_sorted_manh[1],
        dep_uniq_fn_sorted_manh[1], dep_uniq_fn_p_sorted_manh[1],

        dep_euclidean[1], dep_tfidf_euclidean[1], dep_p_euclidean[1],
        dep_uniq_euclidean[1], dep_uniq_tfidf_euclidean[1], dep_uniq_p_euclidean[1],
        dep_uniq_fn_euclidean[1], dep_uniq_fn_tfidf_euclidean[1], dep_uniq_fn_p_euclidean[1],
        dep_bin_euclidean[1], dep_bin_tfidf_euclidean[1],
        dep_sorted_euclidean[1], dep_p_sorted_euclidean[1],
        dep_uniq_fn_sorted_euclidean[1], dep_uniq_fn_p_sorted_euclidean[1],

        dep_cosine[1], dep_tfidf_cosine[1], dep_p_cosine[1],
        dep_uniq_cosine[1], dep_uniq_tfidf_cosine[1], dep_uniq_p_cosine[1],
        dep_uniq_fn_cosine[1], dep_uniq_fn_tfidf_cosine[1], dep_uniq_fn_p_cosine[1],
        dep_bin_cosine[1], dep_bin_tfidf_cosine[1],
        dep_sorted_cosine[1], dep_p_sorted_cosine[1],
        dep_uniq_fn_sorted_cosine[1], dep_uniq_fn_p_sorted_cosine[1],

        fn_r_dot[1], fn_r_tfidf_dot[1],

        dep_r_dot[1], dep_r_bin_dot[1],
        dep_r_tfidf_dot[1], dep_r_bin_tfidf_dot[1],

        dep_r_manh[1], dep_r_bin_manh[1],
        dep_r_tfidf_manh[1], dep_r_bin_tfidf_manh[1],

        dep_r_euclidean[1], dep_r_bin_euclidean[1],
        dep_r_tfidf_euclidean[1], dep_r_bin_tfidf_euclidean[1],

        dep_r_cosine[1], dep_r_bin_cosine[1],
        dep_r_tfidf_cosine[1], dep_r_bin_tfidf_cosine[1],
    ]

    print('Scaling...')
    scl = StandardScaler()
    for i in range(X_train.shape[1]):
        if len(np.unique(X_train[:, i])) > 2:
            scl.fit(X_train[:, i].reshape(-1, 1))
            xtrain = scl.transform(X_train[:, i].reshape(-1, 1)).flatten()
            xtest = scl.transform(X_test[:, i].reshape(-1, 1)).flatten()
            X_train[:, i] = np.clip(xtrain, -25, 25)
            X_test[:, i] = np.clip(xtest, -25, 25)
        else:
            continue

    return X_train, X_test, Y_train, y_train


def get_dot(input_list, y_train):
    X_train, X_test = input_list
    if type(X_train) is sparse.csr.csr_matrix:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    n_samples = X_train.shape[0]
    n_categs = len(np.unique(y_train))
    kfolds = StratifiedKFold(y_train, 4)
    X_train_features = np.zeros([n_samples, n_categs])
    for train, test in kfolds:
        X1 = X_train[train, :]
        y1 = y_train[train]
        X2 = X_train[test, :]
        temp = pd.DataFrame(np.c_[y1.reshape(-1, 1), X1])
        m = np.array(temp.groupby(0).mean())
        X_train_features[test, :] = m.dot(X2.T).T

    temp = pd.DataFrame(np.c_[y_train, X_train])
    m = np.array(temp.groupby(0).mean())
    features_dot = [X_train_features, m.dot(X_test.T).T]
    return features_dot


def get_euclidean(input_list, y_train):
    X_train, X_test = input_list
    if type(X_train) is sparse.csr.csr_matrix:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    n_samples = X_train.shape[0]
    n_categs = len(np.unique(y_train))
    kfolds = StratifiedKFold(y_train, 4)
    X_train_features = np.zeros([n_samples, n_categs])
    for train, test in kfolds:
        X1 = X_train[train, :]
        y1 = y_train[train]
        X2 = X_train[test, :]
        temp = pd.DataFrame(np.c_[y1.reshape(-1, 1), X1])
        m = np.array(temp.groupby(0).mean())
        X_train_features[test, :] = euc_dist(X2, m)

    temp = pd.DataFrame(np.c_[y_train, X_train])
    m = np.array(temp.groupby(0).mean())
    features_euc = [X_train_features, euc_dist(X_test, m)]
    return features_euc


def get_cosine(input_list, y_train):
    X_train, X_test = input_list
    if type(X_train) is sparse.csr.csr_matrix:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    n_samples = X_train.shape[0]
    n_categs = len(np.unique(y_train))
    kfolds = StratifiedKFold(y_train, 4)
    X_train_features = np.zeros([n_samples, n_categs])
    for train, test in kfolds:
        X1 = X_train[train, :]
        y1 = y_train[train]
        X2 = X_train[test, :]
        temp = pd.DataFrame(np.c_[y1.reshape(-1, 1), X1])
        m = np.array(temp.groupby(0).mean())
        X_train_features[test, :] = cos_sim(X2, m)

    temp = pd.DataFrame(np.c_[y_train, X_train])
    m = np.array(temp.groupby(0).mean())
    features_euc = [X_train_features, cos_sim(X_test, m)]
    return features_euc


def get_mahalanobis(input_list, y_train):
    X_train, X_test = input_list
    if type(X_train) is sparse.csr.csr_matrix:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    n_samples = X_train.shape[0]
    n_categs = len(np.unique(y_train))
    kfolds = StratifiedKFold(y_train, 2)
    X_train_features = np.zeros([n_samples, n_categs])

    samples = X_train[np.bool_(y_train == 8), :]
    extra_cov = np.cov(samples, rowvar=False)

    for train, test in kfolds:
        X1 = X_train[train, :]
        y1 = y_train[train]
        X2 = X_train[test, :]
        temp = pd.DataFrame(np.c_[y1.reshape(-1, 1), X1])
        m = np.array(temp.groupby(0).mean())
        C = temp.groupby(0).cov()
        C.loc[8] = extra_cov
        for i in range(m.shape[0]):
            cur_m = m[i, :]
            cur_C = C.loc[i]
            C_inv = np.linalg.pinv(cur_C)

            X2_cen = X2 - cur_m
            mah_dist = np.sqrt(np.sum(
                X2_cen.T * C_inv.dot(X2_cen.T), axis=0).T)
            X_train_features[test, i] = mah_dist

    temp = pd.DataFrame(np.c_[y_train, X_train])
    m = np.array(temp.groupby(0).mean())
    C = temp.groupby(0).cov()
    X_test_features = np.zeros([len(X_test), n_categs])
    for i in range(m.shape[0]):
        cur_m = m[i, :]
        cur_C = C.loc[i]
        C_inv = np.linalg.pinv(cur_C)

        X_test_cen = X_test - cur_m
        mah_dist = np.sqrt(np.sum(
            X_test_cen.T * C_inv.dot(X_test_cen.T), axis=0).T)
        X_test_features[:, i] = mah_dist

    features_maha = [X_train_features, X_test_features]
    return features_maha


def get_manhattan(input_list, y_train):
    X_train, X_test = input_list
    if type(X_train) is sparse.csr.csr_matrix:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    n_samples = X_train.shape[0]
    n_categs = len(np.unique(y_train))
    kfolds = StratifiedKFold(y_train, 4)
    X_train_features = np.zeros([n_samples, n_categs])

    for train, test in kfolds:
        X1 = X_train[train, :]
        y1 = y_train[train]
        X2 = X_train[test, :]
        temp = pd.DataFrame(np.c_[y1.reshape(-1, 1), X1])
        m = np.array(temp.groupby(0).mean())
        for i in range(m.shape[0]):
            cur_m = m[i, :]
            X_train_features[test, i] = np.sum(np.abs(X2 - cur_m), axis=1)

    temp = pd.DataFrame(np.c_[y_train, X_train])
    m = np.array(temp.groupby(0).mean())

    X_test_features = np.zeros([len(X_test), n_categs])
    for i in range(m.shape[0]):
        cur_m = m[i, :]
        X_test_features[:, i] = np.sum(np.abs(X_test - cur_m), axis=1)

    features_dot = [X_train_features, X_test_features]
    return features_dot


def get_knn(input_list, y_train, n_neighbors=301):
    X_train, X_test = input_list
    # pca = PCA(n_components=5)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    if type(X_train) is sparse.csr.csr_matrix:
        X_train = X_train.toarray()
        X_test = X_test.toarray()

    n_samples = X_train.shape[0]
    n_categs = len(np.unique(y_train))
    kfolds = StratifiedKFold(y_train, 2)
    X_train_features = np.zeros([n_samples, n_categs])

    knn = KNN(n_neighbors=n_neighbors)
    for train, test in kfolds:
        X1 = X_train[train, :]
        y1 = y_train[train]
        X2 = X_train[test, :]
        knn.fit(X1, y1)
        X_train_features[test, :] = knn.predict_proba(X2)

    knn.fit(X_train, y_train)
    X_test_features = knn.predict_proba(X_test)

    features_manh = [X_train_features, X_test_features]
    return features_manh


def get_xgb(input_list, y_train, n_folds=2):
    X_train, X_test = input_list

    n_samples = X_train.shape[0]
    n_categs = len(np.unique(y_train))
    kfolds = StratifiedKFold(y_train, n_folds)
    X_train_features = np.zeros([n_samples, n_categs])

    params = {}
    params['eval_metric'] = "mlogloss"
    params['objective'] = "multi:softprob"
    params['num_class'] = len(np.unique(y_train))
    params['silent'] = 1

    params['max_depth'] = 6
    params['min_child_weight'] = 40
    params['eta'] = .3
    params['max_delta_step'] = 1

    n_iters = 0
    for train, test in kfolds:

        X1 = X_train[train, :]
        y1 = y_train[train]
        X2 = X_train[test, :]
        y2 = y_train[test]

        dtrain = xgb.DMatrix(X1, label=y1)
        dtest = xgb.DMatrix(X2, label=y2)

        evallist = [(dtrain, 'train'), (dtest, 'eval')]

        num_round = 200
        num_early = 5
        bst = xgb.train(params, dtrain, num_round, evallist,
                        early_stopping_rounds=num_early)
        n_iters += bst.best_iteration

        X_train_features[test, :] = bst.predict(dtest)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    num_round = int(n_iters / n_folds)
    evallist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_round, evallist)

    X_test_features = bst.predict(dtest)

    features_xgb = [X_train_features, X_test_features]
    return features_xgb
