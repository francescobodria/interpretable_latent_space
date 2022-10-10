#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from exp.data import load_image_dataset, load_tabular_dataset
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from tqdm import tqdm
from scipy.spatial.distance import cdist,euclidean
from growingspheres import counterfactuals as cf
import time
import signal

latent_dimensions = {'credit':25, 'coil2000':10, 'clean1':15, 'clean2':20, 'madelon':7, 'sonar':20}
n_continuous =      {'credit':5,  'coil2000':0,  'clean1':0,  'clean2':0,  'madelon':0, 'sonar':0}

for dataset_name in ['credit',    'coil2000',    'clean1',    'clean2',    'madelon',   'sonar']:
    print(dataset_name)
    
    with open('./results/counterfactual_results.txt', 'a') as f:
        f.write(dataset_name+'\n')
    
    X_train, X_test, Y_train, Y_test, df_old, df  = load_tabular_dataset(dataset_name, './data/datasets/')
    n = 1000
    if X_test.shape[0] < n:
        n = X_test.shape[0]

    class LinearModel(nn.Module):
        def __init__(self, input_shape, latent_dim=2):
            super(LinearModel, self).__init__()

            # encoding components
            self.fc1 = nn.Linear(input_shape, latent_dim)

        def encode(self, x):
            x = self.fc1(x)
            return x

        def forward(self, x):
            z = self.encode(x)
            return z
    
    latent_dim = latent_dimensions[dataset_name]
    model = LinearModel(X_train.shape[1],latent_dim)

    model.load_state_dict(torch.load(f'./models/{dataset_name}_LinearTransparent_latent_{latent_dim}.pt'))

    with torch.no_grad():
        Z_train = model(torch.tensor(X_train,dtype=torch.float32)).numpy()
        Z_test = model(torch.tensor(X_test,dtype=torch.float32)).numpy()

    neigh = KNeighborsClassifier(n_neighbors=25)
    neigh.fit(Z_train, Y_train)
    KNN_pred = neigh.predict(Z_train)
    print(neigh.score(Z_test,Y_test))

    def predict(q):
        return neigh.predict(model(torch.tensor(q).float()).detach())

    def find_changes(start_point, n_continuous, end_point=None, z_end=None, max_iter=5, debug=False):

        if z_end is None:
            z_end   = model(torch.tensor(end_point,dtype=torch.float32)).detach().numpy()
        z_start = model(torch.tensor(start_point,dtype=torch.float32)).detach().numpy()

        w = model.fc1.weight.detach().numpy()
        b = model.fc1.bias.detach().numpy()

        mods = np.zeros(len(start_point[0,:]))

        x_mod = start_point.copy()+mods
        z_mod = model(torch.tensor(x_mod).float()).detach().reshape(1,-1).numpy()

        pred = int(neigh.predict(z_start))
        pred_mod = int(neigh.predict(z_mod))
        j = 1
        dist = distance.euclidean(z_end,z_mod)

        while pred == pred_mod:
            if debug:
                print(distance.euclidean(z_end,z_mod))
            proj = z_end - z_mod
            d = np.argsort(np.abs(proj.dot(w)))
            d_mods = []
            for i in range(-start_point.shape[1],0):
                x_mod_prop = x_mod.copy()
                idx = d[0, i]
                mod = np.sum((z_end-(np.sum(x_mod[0,:idx]*w[:,:idx],axis=1)+np.sum(x_mod[0,idx+1:]*w[:,idx+1:],axis=1))-b)*w[:,idx])/(np.sum(w[:,idx]**2))
                if idx >= n_continuous:
                    mod = np.round(np.clip(mod,-1,1))
                else:
                    mod = np.clip(mod,-1,1)
                x_mod_prop[0, idx] = mod
                z_mod_prop = model(torch.tensor(x_mod_prop).float()).detach().reshape(1,-1).numpy()
                d_mods.append(distance.euclidean(z_end,z_mod_prop))
            idx = d[0,range(-start_point.shape[1],0)[np.argmin(d_mods)]]
            mod = np.sum((z_end-(np.sum(x_mod[0,:idx]*w[:,:idx],axis=1)+np.sum(x_mod[0,idx+1:]*w[:,idx+1:],axis=1))-b)*w[:,idx])/(np.sum(w[:,idx]**2))
            if idx > n_continuous:
                mod = np.round(np.clip(mod,-1,1))
            else:
                mod = np.clip(mod,-1,1)
            mods[idx] += mod - x_mod[0,idx]
            x_mod[0, idx] = mod
            z_mod = model(torch.tensor(x_mod).float()).detach().reshape(1,-1).numpy()

            if distance.euclidean(z_end,z_mod) == dist:
                j +=1
            else:
                j = 1
                dist = distance.euclidean(z_end,z_mod)

            if j == max_iter:
                break
                mods = np.nan

            pred_mod = int(neigh.predict(z_mod))
            if debug:
                print(pred,pred_mod)
                print(j)

        return mods

    if dataset_name == 'madelon':
        centroid_0 = np.mean(Z_train[Y_train==-1],axis=0)
    else:
        centroid_0 = np.mean(Z_train[Y_train==0],axis=0)
    centroid_1 = np.mean(Z_train[Y_train==1],axis=0)

    transp_cf = []
    not_found = []
    for i in tqdm(range(n)):
        q = X_test[i].reshape(1,-1).copy()
        with torch.no_grad():
            z = model(torch.tensor(q,dtype=torch.float32)).numpy()
        if int(neigh.predict(z)):
            mods = find_changes(q, n_continuous[dataset_name], z_end=centroid_0, debug=False)
            if mods is np.nan:
                not_found.append(i)
            else:
                transp_cf.append(mods)
        else:
            mods = find_changes(q, n_continuous[dataset_name], z_end=centroid_1, debug=False)
            if mods is np.nan:
                not_found.append(i)
            else:
                transp_cf.append(mods)
    transp_cf = X_test[:n].copy()+np.vstack(transp_cf)

    X_test_found = np.delete(X_test[:n],not_found,axis=0)
    with open('./results/counterfactual_results.txt', 'a') as f:
        f.write('ILS \n')
        f.write(f'found: {len(X_test_found)/n}\n')
        f.write(f'dis_dist: {np.mean(np.diag(cdist(transp_cf,X_test_found[:n].copy())))}\n')
        f.write(f'dis_count: {np.mean(np.mean(transp_cf!=X_test_found[:n].copy(),axis=1))}\n')
        f.write(f'impl: {np.mean(np.min(cdist(transp_cf,X_test),axis=1))}\n')

    grad_cf = []
    not_found = []
    for i in tqdm(range(n)):
        q = X_test[i].reshape(1,-1).copy()
        loss = torch.nn.MSELoss()
        with torch.no_grad():
            z = model(torch.tensor(q,dtype=torch.float32)).numpy()
        pred = int(neigh.predict(z))
        q = torch.nn.parameter.Parameter(torch.tensor(q, requires_grad=True).float())
        opt = torch.optim.Adam([q],lr=0.001)
        if pred==1:
            centroid = centroid_0
        else:
            centroid = centroid_1
        total_loss = []    
        nf = 0
        while pred == int(neigh.predict(z)):
            loss_epoch = loss(torch.tensor(centroid).reshape(1,-1),model(q)) 
            total_loss.append(loss_epoch.item())
            loss_epoch.backward()
            opt.step()
            opt.zero_grad()
            pred = int(neigh.predict(model(q).detach().numpy()))
            if loss_epoch.item()<1e-10:
                break
                nf = 1
        if nf:
            not_found.append(i)
        else:
            grad_cf.append(q.detach().numpy().ravel())
    
    grad_cf = np.vstack(grad_cf)
    X_test_found = np.delete(X_test[:n],not_found,axis=0)

    with open('./results/counterfactual_results.txt', 'a') as f:
        f.write('Gradient \n')
        f.write(f'found: {len(X_test_found)/n}\n')
        f.write(f'dis_dist: {np.mean(np.diag(cdist(grad_cf,X_test_found[:n].copy())))}\n')
        f.write(f'dis_count: {np.mean(np.mean(grad_cf!=X_test_found[:n].copy(),axis=1))}\n')
        f.write(f'impl: {np.mean(np.min(cdist(grad_cf,X_test),axis=1))}\n')

    class timeout:
        def __init__(self, seconds=1, error_message='Timeout'):
            self.seconds = seconds
            self.error_message = error_message
        def handle_timeout(self, signum, frame):
            raise Exception('timeout')
        def __enter__(self):
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        def __exit__(self, type, value, traceback):
            signal.alarm(0)

    def fit_GS(q, predict):
        CF = cf.CounterfactualExplanation(q, predict, method='GS')
        try:
            with timeout(seconds=15):
                CF.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False)
            return CF
        except:
            CF.enemy = np.nan
            return CF
        
    gs_cf = []
    not_found = []
    for i in tqdm(range(n)):
        start = time.time()
        q = X_test[i].reshape(1,-1).copy()
        CF = fit_GS(q,predict)
        if CF.enemy is np.nan:
            not_found.append(i)
        else:
            gs_cf.append(CF.enemy)
    gs_cf = np.vstack(gs_cf)
    
    X_test_found = np.delete(X_test[:n],not_found,axis=0)
    with open('./results/counterfactual_results.txt', 'a') as f:
        f.write('GS \n')
        f.write(f'found: {len(X_test_found)/n}\n')
        f.write(f'dis_dist: {np.mean(np.diag(cdist(gs_cf,X_test_found[:n].copy())))}\n')
        f.write(f'dis_count: {np.mean(np.mean(gs_cf!=X_test_found[:n].copy(),axis=1))}\n')
        f.write(f'impl: {np.mean(np.min(cdist(gs_cf,X_test),axis=1))}\n')
        f.write('-------------------- \n')
