import numpy as np
import pandas as pd

import numpy as np
import traceback
import random



def cal_threshold(t,p11,p10,p01,p00):
    t1 = 1 / 2 + t / 2 / (p11 + p10)
    t0 = 1 / 2 - t / 2 / (p01 + p00)

    if t1 > 1-1e-5:
        t1 = 1
    elif t1 < 1e-5:
        t1 = 0

    if t0 > 1-1e-5:
        t0 = 1
    elif t0 < 1e-5:
        t0 = 0
    return t1, t0



def cal_t_bound(p11,p10,p01,p00):
    upperbound = min((p11+p10),(p01+p00))
    lowerbound = -1 * upperbound
    return lowerbound, upperbound


def cal_acc(eta, Y, Z, t1, t0):
    Yhat = np.zeros_like(eta)
    Yhat[Z == 1] = (eta[Z == 1] > t1)
    Yhat[Z == 0] = (eta[Z == 0] > t0)
    acc = (Yhat == Y).mean()
    return acc

def cal_disparity(eta,Z,t1,t0):
    disparity = (eta[Z==1]>t1).mean() - (eta[Z==0]>t0).mean()
    return disparity



def fint_thresholds(eta,Y,Z,level=0, pre_level=1e-5):
    p11 = (Z * Y).mean()
    p10 = (Z * (1 - Y)).mean()
    p01 = ((1 - Z) * Y).mean()
    p00 = ((1 - Z) * (1 - Y)).mean()

    lowerbound, upperbound = cal_t_bound(p11,p10,p01,p00)
    disparity0 = cal_disparity(eta,Z,0.5,0.5)
    if abs(disparity0) < level:
        tmid = 0
    elif disparity0 > level:
        tmax = upperbound
        tmin = 0
        level0 = level
        tmid = (tmax + tmin)/2
        while tmax - tmin > pre_level:
            t1, t0 = cal_threshold(tmid,p11,p10,p01,p00)
            disparity = cal_disparity(eta,Z,t1,t0)
            if disparity > level0:
                tmin = tmid
            else:
                tmax = tmid
            tmid = (tmax + tmin)/2
    else:
        tmax = 0
        tmin = lowerbound
        level0 = -1 * level
        tmid = (tmax + tmin)/2
        while tmax - tmin > pre_level:
            t1, t0 = cal_threshold(tmid,p11,p10,p01,p00)
            disparity = cal_disparity(eta,Z,t1,t0)
            if disparity > level0:
                tmin = tmid
            else:
                tmax = tmid
            tmid = (tmax + tmin)/2

    t1, t0 = cal_threshold(tmid,  p11, p10, p01, p00)

    return t1,t0

def number_of_sample(n11,n10,n01,n00,t,n_sample):

    n = n11 + n10 + n01 + n00
    p11 = n11/n
    p10 = n10/n
    p01 = n01/n
    p00 = n00/n
    s11 = p11 * (1 / 2 - t / 2 / (p11 + p10))
    s10 = p10 * (1 / 2 + t / 2 / (p11 + p10))
    s01 = p01 * (1 / 2 + t / 2 / (p01 + p00))
    s00 = p00 * (1 / 2 - t / 2 / (p01 + p00))

    if s11 < 0:
        s11 = 0

    if s10 < 0:
        s10 = 0
    if s01 < 0:
        s01 = 0
    if s00 < 0:
        s00 = 0
    p11new = 0.5 * s11 / (s11 + s10)
    p10new = 0.5 * s10 / (s11 + s10)
    p01new = 0.5 * s01 / (s01 + s00)
    p00new = 0.5 * s00 / (s01 + s00)


    n11_new,n10_new, n01_new, n00_new = round(n_sample * p11new),round(n_sample * p10new), round(n_sample * p01new), round(n_sample * p00new)


    return n11_new,n10_new, n01_new, n00_new






def postprocess(alpha_seed_and_kwargs, postprocessor_factory,
                probas, labels, groups, n_test, n_post,dataset_name):

    if len(alpha_seed_and_kwargs) == 2:
        alpha, seed = alpha_seed_and_kwargs
        kwargs = {}
    else:
        alpha, seed, kwargs = alpha_seed_and_kwargs

  # Split the remaining data into post-processing and test data


    train_probas_post = probas[:n_post]
    train_labels_post = labels[:n_post]
    train_groups_post = groups[:n_post]
    test_probas = probas[n_post:]
    test_labels = labels[n_post:]
    test_groups = groups[n_post:]

    if alpha == np.inf:
    # Evaluate the unprocessed model
        postprocessor = None
        test_preds = test_probas.argmax(axis=1)
    else:
        try:
      # Post-process the predicted probabilities
            postprocessor = postprocessor_factory().fit(train_probas_post,
                                                  train_groups_post,
                                                  alpha=alpha,
                                                  **kwargs)
      # Evaluate the post-processed model
            test_preds = postprocessor.predict(test_probas, test_groups)
        except Exception:
            print(f"Post-processing failed with alpha={alpha} and seed={seed}:\n{traceback.format_exc()}",flush=True)
            data = [seed, dataset_name, alpha, None, None]
            columns = ['seed', 'dataset', 'alpha', 'acc', 'disparity']
            df_test = pd.DataFrame([data], columns=columns)

            return df_test

    acc = (test_preds == test_labels).mean()
    disparity = np.abs((test_preds[test_groups==1]).mean() - (test_preds[test_groups==0]).mean())


    data = [seed,dataset_name,alpha, acc, np.abs(disparity)]
    columns = ['seed','dataset','alpha','acc', 'disparity']

    df_test = pd.DataFrame([data], columns=columns)

    return df_test




def threshold_flipping(pa,eta, Yhat,Y,Z,level):

    s = ((1-Z)/(1-pa) - Z/pa) * (2* Yhat-1) /   ( (2 * eta - 1)*(2 * Yhat-1))
    ssort = s.argsort()
    n1 = Yhat[Z==1].sum()
    n0 = Yhat[Z==0].sum()
    acc = (Yhat == Y).mean()
    acc_max = 0
    tstar = -100000
    n = len(Z)
    p1 = n1/n
    p0= n0/n
    dpstar  = -199
    for idx in ssort:
        t = s[idx]

        acc = acc + (1-2*(Yhat[idx]==Y[idx]))/n
        if Z[idx]==1:
            p1 = p1 + (1- 2 * Yhat[idx])/n
        if Z[idx]==0:
            p0 = p0 + (1- 2 * Yhat[idx])/n
        dp = np.abs(p1 / pa - p0/(1-pa))
        if (dp<=level) & (acc>acc_max):
            tstar = t
            acc_max = acc
            dpstar = dp


    return tstar





def postpreocessing_flipping(pa,eta, Yhat,Z,t):

    s = ((1-Z)/(1-pa) - Z/pa) * (2* Yhat-1) /     ( (2 * eta - 1)*(2 * Yhat-1))
    Yhat_new = (s/t<=1) * Yhat  +    (s/t>1) * (1- Yhat)

    return Yhat_new



def cal_acc_PPF(Yhat, Y):

    acc = (Yhat == Y).mean()
    return acc

def cal_disparity_PPF(Yhat,Z):


    disparity = (Yhat[Z==1]).mean() - (Yhat[Z==0]).mean()

    return disparity



def sample_points(dataset, alldata, number):
    idxs = []
    s_lists = []

    for t in range(number):
        s = -1
        while s not in alldata.index:
            idx = random.choices(dataset.index, weights=dataset['weight'])[0]
            points_to_selects = dataset.loc[idx, :]['neighbour']
            s = int(random.choices(points_to_selects.strip('[').strip(']').split(','))[0])
        idxs.append(idx)
        s_lists.append(s)
    data0 = dataset.loc[idxs, :].drop(['neighbour', 'weight'], axis=1)
    data1 = alldata.loc[s_lists, :].drop(['neighbour', 'weight'], axis=1)
    beta = np.random.random((number, 1))

    new_data = data0.values * (1 - beta) + data1.values * beta

    columns = data0.keys()
    syndataset = pd.DataFrame(new_data, columns=columns)

    return syndataset






