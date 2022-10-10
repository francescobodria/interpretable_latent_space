#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

import torch
random_seed = 42
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

from exp.data import load_tabular_dataset, load_image_dataset
from exp.models import *
from exp.evaluation import compute_metrics
import pickle
    
for dataset_name in ['MNIST','FashionMNIST']: 
    print(f'evaluating {dataset_name}')
    dataset_path = './data'
    X_train, X_test, Y_train, Y_test, train_loader, test_loader = load_image_dataset(dataset_name, dataset_path)
    latent_dims = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30]
    
    for i in latent_dims:
        print(f'latent dimension: {i}')
        result = {'transparent':{},
             'vae':{},
             'pca':{},
             'tsne':{},
             'umap':{},
             'trimap':{}
             }

        start = time.time()
        d = linear_transparent_eval(X_train, X_test, i, dataset_name=dataset_name, max_epochs=10)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['transparent'] = d

        start = time.time()
        d = vae_eval(X_train, X_test, i, dataset_name=dataset_name)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['vae'] = d

        start = time.time()
        d = pca_eval(X_train, X_test, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['pca'] = d

        start = time.time()
        d = tsne_eval(X_train, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['tsne'] = d

        start = time.time()
        d = umap_eval(X_train, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['umap'] = d

        start = time.time()
        d = trimap_eval(X_train, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['trimap'] = d

        pickle.dump(result,open(f'./results/{dataset_name}_latent_{i}.p','wb'))
        upload_files(f'./results/{dataset_name}_latent_{i}.p', S3_BUCKET_NAME)
        print('-----------------------------------------')
        
    print('*************************************')

names = ['credit', 'adult', 'cover', 'coil2000', 'clean1', 'clean2', 'isolet', 'madelon', 'sonar', 'soybean', 'anneal']
for dataset_name in names: 
    print(f'evaluating {dataset_name}')
    dataset_path = './data/datasets'
    X_train, X_test, Y_train, Y_test = load_tabular_dataset(dataset_name, dataset_path)
    latent_dims = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30]
    
    for i in latent_dims:
        print(f'latent dimension: {i}')
        result = {'transparent':{},
             'vae':{},
             'pca':{},
             'tsne':{},
             'umap':{},
             'trimap':{}
             }

        print('evaluating transparent')
        start = time.time()
        d = linear_transparent_eval(X_train, X_test, i, dataset_name=dataset_name, max_epochs=10)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['transparent'] = d

        print('evaluating VAE')
        start = time.time()
        d = vae_eval(X_train, X_test, i, dataset_name=dataset_name)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['vae'] = d

        start = time.time()
        d = pca_eval(X_train, X_test, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['pca'] = d

        start = time.time()
        d = tsne_eval(X_train, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['tsne'] = d

        start = time.time()
        d = umap_eval(X_train, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['umap'] = d

        start = time.time()
        d = trimap_eval(X_train, i)
        d['metrics'] = compute_metrics(X_train, d['Z_train'], Y_train, n_jobs=-1)
        d['metrics']['running_time'] = time.time()-start
        result['trimap'] = d

        pickle.dump(result,open(f'./results/{dataset_name}_latent_{i}.p','wb'))
        
        print('-----------------------------------------')
    print('*************************************')

