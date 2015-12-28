from __future__ import print_function
from __future__ import division

import random

import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedKFold

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.constraints import maxnorm

import xgboost as xgb


def load_data():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    return df_train, df_test


def relabel_columns(df_train, df_test):
    sel_phx = np.logical_and(
        df_train.DepartmentDescription == 'PHARMACY RX',
        np.isnan(df_train.FinelineNumber))
    sel_nphx = np.logical_and(
        df_train.DepartmentDescription != 'PHARMACY RX',
        np.isnan(df_train.FinelineNumber))

    df_train['Upc'][sel_phx] = -1
    df_train['FinelineNumber'][sel_phx] = -1
    df_train['Upc'][sel_nphx] = 0
    df_train['FinelineNumber'][sel_nphx] = 0
    df_train['DepartmentDescription'][sel_nphx] = 'Unknown'

    sel_phx = np.logical_and(
        df_test.DepartmentDescription == 'PHARMACY RX',
        np.isnan(df_test.FinelineNumber))
    sel_nphx = np.logical_and(
        df_test.DepartmentDescription != 'PHARMACY RX',
        np.isnan(df_test.FinelineNumber))

    df_test['Upc'][sel_phx] = -1
    df_test['FinelineNumber'][sel_phx] = -1
    df_test['Upc'][sel_nphx] = 0
    df_test['FinelineNumber'][sel_nphx] = 0
    df_test['DepartmentDescription'][sel_nphx] = 'Unknown'

    # Sequentially rename TripType
    relabel_tt = {}
    column_names = ['VisitNumber']
    ctr = 0
    for i in np.sort(np.unique(df_train['TripType'])):
        relabel_tt[i] = ctr
        column_names.append('TripType_' + str(i))
        ctr += 1
    df_train['TripType_'] = df_train['TripType'].apply(lambda x: relabel_tt[x])

    # Sequentially rename VisitNumber for df_train
    relabel_vn = {}
    ctr = 0
    for i in np.sort(np.unique(df_train['VisitNumber'])):
        relabel_vn[i] = ctr
        ctr += 1
    df_train['VisitNumber_'] = df_train['VisitNumber'].apply(
        lambda x: relabel_vn[x])

    # Sequentially rename VisitNumber for df_test
    relabel_vn = {}
    ctr = 0
    for i in np.sort(np.unique(df_test['VisitNumber'])):
        relabel_vn[i] = ctr
        ctr += 1
    df_test['VisitNumber_'] = df_test['VisitNumber'].apply(
        lambda x: relabel_vn[x])

    # Sequentially rename Upc number
    relabel_upc = {}
    ctr = 0
    for i in np.sort(np.unique(np.concatenate(
            [df_train['Upc'], df_test['Upc']]))):
        relabel_upc[i] = ctr
        ctr += 1
    df_train['Upc_'] = df_train['Upc'].apply(
        lambda x: relabel_upc[x])
    df_test['Upc_'] = df_test['Upc'].apply(
        lambda x: relabel_upc[x])

    # Sequentially rename FinelineNumber
    relabel_fn = {}
    ctr = 0
    for i in np.sort(np.unique(np.concatenate(
            [df_train['FinelineNumber'], df_test['FinelineNumber']]))):
        relabel_fn[i] = ctr
        ctr += 1
    df_train['FinelineNumber_'] = df_train['FinelineNumber'].apply(
        lambda x: relabel_fn[x])
    df_test['FinelineNumber_'] = df_test['FinelineNumber'].apply(
        lambda x: relabel_fn[x])

    # Sequentially rename DepartmentDescription
    relabel_dd = {}
    ctr = 0
    for i in np.sort(np.unique(np.concatenate([
            df_train['DepartmentDescription'],
            df_test['DepartmentDescription']]))):
        relabel_dd[i] = ctr
        ctr += 1
    df_train['Department_'] = df_train['DepartmentDescription'].apply(
        lambda x: relabel_dd[x])
    df_test['Department_'] = df_test['DepartmentDescription'].apply(
        lambda x: relabel_dd[x])

    # Sequentially rename Weekday
    relabel_wd = {}
    ctr = 0
    for i in np.sort(np.unique(np.concatenate([
            df_train['Weekday'],
            df_test['Weekday']]))):
        relabel_wd[i] = ctr
        ctr += 1
    df_train['Weekday_'] = df_train['Weekday'].apply(
        lambda x: relabel_wd[x])
    df_test['Weekday_'] = df_test['Weekday'].apply(
        lambda x: relabel_wd[x])

    # a few additional
    df_train['ScanCount_rect'] = df_train['ScanCount'].apply(
        lambda x: x if x > 0 else 0
    )
    df_test['ScanCount_rect'] = df_test['ScanCount'].apply(
        lambda x: x if x > 0 else 0
    )

    df_train['ScanCount_rect_neg'] = df_train['ScanCount'].apply(
        lambda x: -x if x < 0 else 0
    )
    df_test['ScanCount_rect_neg'] = df_test['ScanCount'].apply(
        lambda x: -x if x < 0 else 0
    )

    df_train['ScanCount_binary'] = df_train['ScanCount'].apply(
        lambda x: 1 if x > 0 else 0
    )
    df_test['ScanCount_binary'] = df_test['ScanCount'].apply(
        lambda x: 1 if x > 0 else 0
    )

    df_train['ScanCount_binary_neg'] = df_train['ScanCount'].apply(
        lambda x: 1 if x < 0 else 0
    )
    df_test['ScanCount_binary_neg'] = df_test['ScanCount'].apply(
        lambda x: 1 if x < 0 else 0
    )


def train_nn(X_train, X_test, Y_train):

    nb_folds = 4
    y_train = np.argmax(Y_train, axis=1)
    train, valid = list(StratifiedKFold(y_train, nb_folds))[0]
    n_categs = Y_train.shape[1]
    f = 0
    avg_logloss = 0
    folds = True
    if folds:
        for dim in [756]:
            print('---' * 20)
            print('Fold', dim)
            print('---' * 20)
            f += 1
            X_train_subset = X_train[train]
            X_valid_subset = X_train[valid]
            Y_train_subset = Y_train[train]
            Y_valid_subset = Y_train[valid]

            print("Building model...")

            model = Sequential()
            model.add(Dense(output_dim=dim, input_dim=X_train.shape[1],
                            init='glorot_uniform',
                            W_constraint=maxnorm(.95)))

            model.add(PReLU())
            model.add(Dropout(0.6))

            model.add(Dense(output_dim=n_categs, init='glorot_uniform',
                            W_constraint=maxnorm(.95)))

            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer='adagrad')
            model.fit(X_train_subset, Y_train_subset,
                      nb_epoch=500, batch_size=512,
                      validation_data=(X_valid_subset, Y_valid_subset))

            valid_preds = model.predict_proba(X_valid_subset, verbose=0)
            return valid_preds

    print('---' * 20)
    print('Avg log loss: {}'.format(float(avg_logloss / nb_folds)))
    print('---' * 20)

    model = Sequential()

    model.add(Dense(output_dim=756, input_dim=X_train.shape[1],
                    init="glorot_uniform", W_constraint=maxnorm(0.95)))
    model.add(PReLU())
    model.add(Dropout(0.6))

    model.add(Dense(output_dim=n_categs, init="glorot_uniform",
                    W_constraint=maxnorm(0.95)))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad')

    print("Training model...")
    model.fit(X_train, Y_train, nb_epoch=400, batch_size=128)

    #########################################################################

    probs = model.predict_proba(X_test, batch_size=128)

    column_names, index = get_extra()

    df_out = pd.DataFrame(data=np.c_[index, probs], columns=column_names)
    df_out['VisitNumber'] = np.int32(df_out['VisitNumber'])
    df_out.to_csv('nn.csv', index=False)
    return df_out


def make_submission(probs):

    column_names, index = get_extra()
    df_out = pd.DataFrame(data=np.c_[index, probs], columns=column_names)
    df_out['VisitNumber'] = np.int32(df_out['VisitNumber'])
    df_out.to_csv('nn.csv', index=False)
    return df_out


def train_ensemble(X_train, X_test, Y_train, n_models=6):
    n_categs = Y_train.shape[1]
    for i in range(n_models):
        print('---' * 20)
        print('Training model #: {}'.format(i + 1))
        print('---' * 20)

        model = Sequential()

        dim = random.choice(np.arange(512, 769))
        model.add(Dense(output_dim=dim, input_dim=X_train.shape[1],
                        init="glorot_uniform", W_constraint=maxnorm(1)))
        model.add(PReLU())
        model.add(Dropout(0.6))

        model.add(Dense(output_dim=n_categs, init="glorot_uniform",
                        W_constraint=maxnorm(1)))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adagrad')
        print("Training model...")

        epoch = random.choice(np.arange(100, 400))

        model.fit(X_train, Y_train, nb_epoch=epoch, batch_size=512)

        if i == 0:
            probs = model.predict_proba(X_test, batch_size=512)
        else:
            probs += model.predict_proba(X_test, batch_size=512)

    probs /= n_models
    column_names, index = get_extra()
    df_out = pd.DataFrame(data=np.c_[index, probs], columns=column_names)
    df_out['VisitNumber'] = np.int32(df_out['VisitNumber'])
    df_out.to_csv('nnEnsemble.csv', index=False)
    return df_out


def train_xgb(X_train, X_test, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    evallist = [(dtrain, 'train')]

    params = {}
    params['eval_metric'] = "mlogloss"
    params['objective'] = "multi:softprob"
    params['num_class'] = len(np.unique(y_train))
    params['silent'] = 1

    params['max_depth'] = 6
    params['min_child_weight'] = 60
    params['eta'] = .1
    params['max_delta_step'] = 1

    # num_round = 120
    num_round = 275
    bst = xgb.train(params, dtrain, num_round, evallist)
    probs = bst.predict(dtest)

    column_names, index = get_extra()
    df_out = pd.DataFrame(data=np.c_[index, probs], columns=column_names)
    df_out['VisitNumber'] = np.int32(df_out['VisitNumber'])
    df_out.to_csv('xgb.csv', index=False)
    return df_out


def get_extra():
    df_train, df_test = load_data()

    column_names = ['VisitNumber']
    for i in np.sort(np.unique(df_train['TripType'])):
        column_names.append('TripType_' + str(i))

    index = np.zeros(len(np.unique(df_test['VisitNumber'])))
    ctr = 0
    for i in np.sort(np.unique(df_test['VisitNumber'])):
        index[ctr] = i
        ctr += 1

    return column_names, index
