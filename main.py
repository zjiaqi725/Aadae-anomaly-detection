# -*- coding: utf-8 -*-
"""
@Project: ADCoder20201118
@File:    AADAE
@Author:  Jiaqi
@Time:    2020/11/18 18:30
@Description: The AADAE for Unsupervised Anomaly Detection,
feature clustering while doing anomaly detection,
could be trained under weakly-supervised (anomaly-free) and unsupervised
"""

import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,auc,precision_recall_curve
from models import ADModule
from sklearn.manifold import TSNE
from datasets.dataset import DATASET
from utils import compute_metrics
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import pickle
import random


class MAutoencoder(nn.Module):
    def __init__(self, name,dataset_name,lan_dim,z_dim,shrink_thres=0.0025):
        super(MAutoencoder, self).__init__()
        self.name = name
        if dataset_name == 'thyroid':     
            self.encoder = nn.Sequential(
                    nn.Linear(6, 4),
                    nn.ReLU(True),
                    nn.Linear(4, 3),
                    nn.ReLU(True),
                    nn.Linear(3, z_dim))
        elif dataset_name == 'mnist':     
            self.encoder = nn.Sequential(
                    nn.Linear(100, 64),
                    nn.ReLU(True),
                    nn.Linear(64, 16),
                    nn.ReLU(True),
                    nn.Linear(16, z_dim))
        elif dataset_name == 'pima':   
            self.encoder = nn.Sequential(
                    nn.Linear(8, 4),
                    nn.ReLU(True),
                    nn.Linear(4, 3),
                    nn.ReLU(True),
                    nn.Linear(3, z_dim))     
        elif dataset_name == 'arrhythmia':   
            self.encoder = nn.Sequential(
                nn.Linear(274, 128),
                nn.ReLU(True),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, z_dim))
            
        elif dataset_name == 'campaign':   
            self.encoder = nn.Sequential(
                nn.Linear(42, 20),
                nn.ReLU(True),
                nn.Linear(20, 6),
                nn.ReLU(True),
                nn.Linear(6, z_dim))
        elif dataset_name == 'satellite':     
            self.encoder = nn.Sequential(
                nn.Linear(36, 18),
                nn.ReLU(True),
                nn.Linear(18, 8),
                nn.ReLU(True),
                nn.Linear(8, z_dim))

        elif dataset_name == 'wbc':     
            self.encoder = nn.Sequential(
                nn.Linear(30, 18),
                nn.ReLU(True),
                nn.Linear(18, 8),
                nn.ReLU(True),
                nn.Linear(8, z_dim))
        self.lan_rep = ADModule(lan_dim=lan_dim, fea_dim=z_dim, shrink_thres =shrink_thres)    
        if dataset_name == 'thyroid': 
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 3),
                nn.ReLU(True),
                nn.Linear(3, 4),
                nn.ReLU(True),
                nn.Linear(4, 6),
                nn.Tanh())    
        elif dataset_name == 'mnist': 
            self.decoder = nn.Sequential(
                 nn.Linear(z_dim, 8),
                 nn.ReLU(True),
                 nn.Linear(8, 32),
                 nn.ReLU(True),
                 nn.Linear(32, 64),
                 nn.ReLU(True),
                 nn.Linear(64, 100),
                 nn.Tanh())
        elif dataset_name == 'pima': 
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 3),
                nn.ReLU(True),
                nn.Linear(3, 4),
                nn.ReLU(True),
                nn.Linear(4, 8),
                nn.Tanh() )
        elif dataset_name == 'arrhythmia': 
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 64),
                nn.ReLU(True),
                nn.Linear(64, 128),
                nn.ReLU(True),
                nn.Linear(128, 274),
                nn.Tanh())
        elif dataset_name == 'campaign':   
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 6),
                nn.ReLU(True),
                nn.Linear(6, 20),
                nn.ReLU(True),
                nn.Linear(20, 42),
                nn.Tanh())
        elif dataset_name == 'satellite': 
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 8),
                nn.ReLU(True),
                nn.Linear(8, 18),
                nn.ReLU(True),
                nn.Linear(18, 36),
                nn.Tanh())
        elif dataset_name == 'wbc': 
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 8),
                nn.ReLU(True),
                nn.Linear(8, 18),
                nn.ReLU(True),
                nn.Linear(18, 30),
                nn.Tanh())

    def forward(self, x):
        x = x.float()
        z1 = self.encoder(x)
        res_lan = self.lan_rep(z1)
        z = res_lan['output']
        att = res_lan['att']    
        lan = res_lan['lan']    #all landmarks  n*z_dim 
        l = res_lan['landmark_nearest']
        l2 = res_lan['landmark_nearest2']
        xhat = self.decoder(z)
        col = res_lan['col']
        lan = res_lan['lan']
        return {'output': xhat , 'latent_o': z1  , 'landmark': lan ,'latent_lanz': z , 'att': att, 'landmark_nearest': l, 'landmark_nearest2': l2, 'col':col }   

def entropy(labels):
    prob_dict = {x:labels.count(x)/len(labels) for x in labels}
    probs = np.array(list(prob_dict.values()))
    return - probs.dot(np.log2(probs))


def train(model, num_epochs, X, batch_size ):
    
    train_loader = DataLoader(X, batch_size, shuffle=True)
    losses = np.zeros(num_epochs)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)
    for epoch in range(num_epochs):
        i = 0
        for img in train_loader:
            x = img.view(img.size(0), -1).float()
            if cuda:
                x = Variable(x).cuda()
            else:
                x = Variable(x)
            hat = model(x)
            xhat = hat['output']
        
            #distillation
            col = hat['col']
            xhh = []
            xh = []
            for i in col:
                xhh.append(xhat[i].detach().numpy())
                xh.append(xhat[i].detach().numpy())
            xh=torch.tensor(xh)
            xhh=torch.tensor(xhh)
            l4 = mse_loss(xh, xhh)
            lhat = torch.Tensor(hat['landmark_nearest'])
            lhat2 = torch.Tensor(hat['landmark_nearest2'])
            c2 = hat['latent_lanz']
            #loss
            l1 = mse_loss(xhat, x)
            l2 = mse_loss(c2, lhat)
            l3 = mse_loss(c2, lhat)-mse_loss(c2, lhat2)
            loss = l1 + alpha*l2 + beta*l3-l4
            losses[epoch] = losses[epoch] * (i / (i + 1.)) + loss * (1. / (i + 1.))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        print('epoch [{}/{}], loss: {:.4f}'.format(
                epoch + 1,
                num_epochs,
                loss))
            
def predict(model, X):   
    for img in X:
        x = img.view(img.size(0), -1)
        if cuda:
            x = Variable(x).cuda()
        else:
            x = Variable(x)
        res = model(x)
        xhat = res['output']
        latent_o = res['latent_o']
        latent_lanz = res['latent_lanz']
        att = res['att']
        landmark = res['landmark']
        x = x.cpu().detach().numpy()
        xhat = xhat.cpu().detach().numpy()
        att = att.cpu().detach().numpy().tolist()

    recon_err = np.zeros(len(x))  
    for i in range(len(x)):
        recon_err[i] = np.sum(np.abs(x[i] - xhat[i])) 
    return {'output': recon_err , 'landmark':landmark  ,'latent_o':latent_o  , 'latent_lanz': latent_lanz , 'att':att } 


def get_result(y_test, recon_err_test, n_lin=200, decimal=4, verbose=False):
    """   Compute the metrics for anomaly detection results   """
    threshold = np.linspace(min(recon_err_test),max(recon_err_test), n_lin)
    acc_list = []
    f1_list = []
    auc_list = []
    for t in threshold:
        y_pred = (recon_err_test>t).astype(np.int)
        acc_list.append(accuracy_score(y_pred,y_test))
        f1_list.append(f1_score(y_pred,y_test))
        auc_list.append(roc_auc_score(y_test,y_pred ))
    
    i = np.argmax(auc_list)   
    t = threshold[i]
    score = f1_list[i]
    print('Recommended threshold: %.3f, related f1 score: %.3f'%(t,score))
    y_pred = (recon_err_test>t).astype(np.int)
    print('\nTest set : AUC_ROC: {:.3f}  F1:{:.3f}  Accuracy:{:.3f}'.format(roc_auc_score(y_test,recon_err_test ),f1_score(y_pred,y_test) ,accuracy_score(y_pred,y_test))) 
    FN = ((y_test==1) & (y_pred==0)).sum()
    FP = ((y_test==0) & (y_pred==1)).sum()
    TP = ((y_test==1) & (y_pred==1)).sum()
    print('precision: {:.4f}'.format(TP/(TP+FP)))
    print('Recall: {:.4f}'.format(TP/(FN+TP)))
#    print('precision: {:.4f}'.format(precision))
#    print('Recall: {:.4f}'.format(recall))

    precision, recall, _thresholds = precision_recall_curve(y_test, recon_err_test)
    area = auc(recall, precision)
    print('AUC_PR: {:.4f}'.format(area))
    
    metrics_dict = {}
    metrics_dict['accuracy'] = round(accuracy_score(y_pred,y_test), decimal)
    metrics_dict['precision'] = round(TP/(TP+FP), decimal)
    metrics_dict['recall'] = round(TP/(FN+TP), decimal)
    metrics_dict['f_score'] = round(f1_score(y_pred,y_test), decimal)
    metrics_dict['auc_roc'] = round(roc_auc_score(y_test,recon_err_test), decimal)
    metrics_dict['auc_pr'] = round(auc(recall, precision), decimal)
    if verbose:
        print(pd.DataFrame(metrics_dict, index=['Results']))
    return metrics_dict

class ExperimentNdNmNr():
    """Experiment class for multiple datasets, multiple models, multiple runs"""

    def __init__(self, datasets: list, models: list, num_runs: int, seed):
        self.datasets = datasets
        self.models = models
        self.num_runs = num_runs
        self.seed = seed
        self.name = '_'.join(['run' + str(self.num_runs)] + \
                             [ds.name for ds in self.datasets] + \
                             [model.name for model in self.models])   
        
    def onerun(self, model, dataset_name, batch_size):
        """Given a model and a dataset, to do once experiment"""
        
        Dataset = DATASET(name=dataset_name, scalertype='StandarScaler')
        X_train, X_test, y_train, y_test = Dataset.weaksupervised_split(7)
    
        train(model, num_epochs, X_train, batch_size ) 

        test_loader = DataLoader(X_test, batch_size=len(X_test), shuffle=False)   
        recon_err_test = predict(model, test_loader)['output']    #####
        
        #anomaly score
        Ent = []
        for i in predict(model, test_loader)['att']:
            Ent.append(lam *entropy(i))
        recon_err_test = recon_err_test +  Ent
        metrics = get_result(np.array(y_test), recon_err_test)
        return metrics
    
    def allruns(self, num_epochs, debugfailexp=False):
        """Do all the experiments for all datasets and models"""
        print('*' * (len(self.name) + 8))
        print(f'**  {self.name}  **')
        print('*' * (len(self.name) + 8))

        results_df = pd.DataFrame({'Dataset': [],
                                   'Model': [],
                                   'irun': [],
                                   'auc_roc': [],
                                   'auc_pr': [],
                                   'f1_score': [],
                                   'recall': []})
        failed_experiments = []
        for ds in self.datasets:
            for model in self.models:
                for irun in range(1, self.num_runs + 1):
#                    random_state = np.random.RandomState(irun)
                    print('-' * 60)
                    print(f'Starting experiment on Dataset:{ds.name}  Det:{model.name}  irun:{irun}')

                    # self.onerun(model, ds, random_state)
                    if debugfailexp == True:
                        self.onerun(model, ds.name, batch_size)
                    else:
                        try:
                            if model == AADAE:
                                model_aadae = MAutoencoder('AADAE_', str(ds.name), lan_dim_in, z_dim)
                                metrics_dict = self.onerun(model_aadae, ds.name, batch_size)
                                results_df = results_df.append({'Dataset': ds.name,
                                                                'Model': model.name,
                                                                'irun': irun,
                                                                'auc_roc': metrics_dict['auc_roc'],
                                                                'auc_pr': metrics_dict['auc_pr'],
                                                                'f1_score': metrics_dict['f_score'],
                                                                'recall': metrics_dict['recall']}, ignore_index=True)
                            else:
                                metrics_dict = self.onerun(model, ds.name, batch_size)
                                results_df = results_df.append({'Dataset': ds.name,
                                                                'Model': model.name,
                                                                'irun': irun,
                                                                'auc_roc': metrics_dict['auc_roc'],
                                                                'auc_pr': metrics_dict['auc_pr'],
                                                                'f1_score': metrics_dict['f_score'],
                                                                'recall': metrics_dict['recall']}, ignore_index=True)
                                
                        except:
                            print(f'FAILED EXPERIMENT ON Dataset:{ds.name}  Det:{model.name}  irun:{irun}')
                            failed_experiments.append(f'Dataset:{ds.name}  Det:{model.name}  irun:{irun}')
        results_df.to_csv(str(num_epochs) + self.name + '.csv')
        resultTable = tabulate(results_df, headers='keys', tablefmt='psql')
        print(resultTable)

    def processResult(self, num_epochs, csvfile=None):
        if csvfile is None:
            df = pd.read_csv(str(num_epochs) +self.name + '.csv', index_col=0)
        elif csvfile is not None:
            df = pd.read_csv(csvfile + '.csv')

        mean_df = df.groupby(['Dataset', 'Model'], as_index=False)['auc_roc', 'auc_pr', 'f1_score','recall'].mean()
        var_df = df.groupby(['Dataset', 'Model'], as_index=False)['auc_roc', 'auc_pr', 'f1_score'].var()
        df = pd.DataFrame()
        df['Dataset'] = mean_df['Dataset']
        df['Model'] = mean_df['Model']
#        df['mode'] = mean_df['mode']
        df['mean_roc'] = mean_df['auc_roc'].round(3)   
        df['mean_pr'] = mean_df['auc_pr'].round(3)
        df['mean_f1'] = mean_df['f1_score'].round(3)
        
        df['var_roc'] = var_df['auc_roc'].round(3)
        df['var_pr'] = var_df['auc_pr'].round(3)
        df['var_f1'] = var_df['f1_score'].round(3)
        
        df['recall'] = mean_df['recall'].round(3)
        
        if csvfile is None:
            df.to_csv(str(num_epochs) + 'MeanResult_' + self.name + '.csv')
        elif csvfile is not None:
            df.to_csv(str(num_epochs) + 'MeanResult_' + csvfile + '.csv')
        print(tabulate(df, headers='keys', tablefmt='psql'))


def build_Datasets():
    return [
        DATASET(name="arrhythmia", scalertype='MinMaxScaler', semirate=0.1, test_size=0.2),
        DATASET(name="campaign", scalertype='MinMaxScaler', semirate=0.1, test_size=0.2),
        DATASET(name="mnist", scalertype='MinMaxScaler', semirate=0.1, test_size=0.2),
        DATASET(name="pima", scalertype='MinMaxScaler', semirate=0.1, test_size=0.2),
        DATASET(name="satellite", scalertype='MinMaxScaler', semirate=0.1, test_size=0.2),
        DATASET(name="thyroid", scalertype='MinMaxScaler', semirate=0.1, test_size=0.2),
        DATASET(name="wbc", scalertype='MinMaxScaler', semirate=0.1, test_size=0.2)
    ]

#define hyperparameter
z_dim = 2
batch_size = 16  
num_epochs =10   
learning_rate = 6.0e-4 
cuda = False
lan_dim_in = 10
alpha = 0.1
beta = 0.01  
lam = 0.3

## thyroid,mnist,pima,arrhythmia,campaign,satellite,musk
dataset_name = 'arrhythmia'

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))  
])

datasets = build_Datasets()
AADAE = MAutoencoder('AADAE_', dataset_name, lan_dim_in, z_dim)

models = [AADAE]
if cuda:
    for model in models:
        model.cuda()
experiment = ExperimentNdNmNr(datasets, models, num_runs=10,  seed=2)
#experiment.onerun(AADAE, dataset_name, batch_size)
experiment.allruns(num_epochs)
experiment.processResult(num_epochs)
    
    
    

