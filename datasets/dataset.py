 # encoding: utf-8

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate
import h5py
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import pairwise_distances
import sys,os

data_dir = "./datasets"

class DATASET:
    def __init__(self, name, scalertype='StandarScaler', semirate=0.5, test_size=0.2):
        self.filepath = data_dir
        self.name = name
        self.X = None
        self.y = None
        self.dim = 0
        self.numSample = 0
        self.Download()
        self.rate = np.sum(self.y)/float(self.numSample)
        self.semirate = semirate
        self.test_size = test_size
        if scalertype =='StandarScaler':
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
        elif scalertype == 'MinMaxScaler':
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X)

    def Download(self):
        if self.name == 'mnist':   
            data = sio.loadmat(self.filepath+"/mnist/mnist.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape
        elif self.name == 'thyroid':  
            data = sio.loadmat(self.filepath+"/thyroid/thyroid.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape
        elif self.name == 'fraud_noamount':
            data = pd.read_csv(self.filepath+"/creditcardfraud/creditcard.csv")
            self.X = np.array(data.iloc[:10000,1:-2])
            self.y = np.array(data.iloc[:10000,-1])
            self.numSample, self.dim = self.X.shape
        elif self.name == 'fraud_withamount':
            data = pd.read_csv(self.filepath+"/creditcardfraud/creditcard.csv")
            self.X = np.array(data.iloc[:10000,1:-1])
            self.y = np.array(data.iloc[:10000,-1])
            self.numSample, self.dim = self.X.shape
        elif self.name == 'campaign':   ##
            data = pd.read_csv(self.filepath+"/campaign/bank_onehot.csv")
            self.X = np.array(data.iloc[:, 0:-1])
            self.y = np.array(data.iloc[:, -1])
            self.numSample, self.dim = self.X.shape
        elif self.name == 'arrhythmia':   ##
            data = sio.loadmat(self.filepath+"/arrhythmia/arrhythmia.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape
            
        elif self.name == 'kddcup99':
            with h5py.File(self.filepath+"/kddcup99/http.mat", 'r') as data:
                self.X = np.array(data['X']).T
                self.y = np.array(data['y']).T
                self.numSample, self.dim = self.X.shape

        elif self.name == 'pima':   ##
            data = sio.loadmat(self.filepath+"/pima/pima.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'glass':
            data = sio.loadmat(self.filepath+"/glass/glass.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'letter':
            data = sio.loadmat(self.filepath + "/letter/letter.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'ionosphere':
            data = sio.loadmat(self.filepath + "/ionosphere/ionosphere.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'wbc':
            data = sio.loadmat(self.filepath + "/wbc/wbc.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'vowels':
            data = sio.loadmat(self.filepath + "/vowels/vowels.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'vertebral':
            data = sio.loadmat(self.filepath + "/vertebral/vertebral.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'satellite':  ##
            data = sio.loadmat(self.filepath + "/satellite/satellite.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'lympho':
            data = sio.loadmat(self.filepath + "/lympho/lympho.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'musk':
            data = sio.loadmat(self.filepath + "/musk/musk.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

        elif self.name == 'shuttle':
            data = sio.loadmat(self.filepath + "/shuttle/shuttle.mat")
            self.X = data['X']
            self.y = data['y']
            self.numSample, self.dim = self.X.shape

    def Semi_split(self, random_state):
        # random_state = 1
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=self.test_size, random_state=random_state) # random split the training set and test set
        outlier_indices = np.where(y_train == 1)[0]
        num_outlier = len(outlier_indices)
        y_train_semi = np.zeros_like(y_train)   
        y_train_semi[np.random.choice(outlier_indices, round(num_outlier*self.semirate), replace=False)] = 1
        y_train_total = y_train
        return X_train, X_test, y_train_semi, y_train_total, y_test

    def weaksupervised_split(self, random_state):
        # random_state = 1   
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=self.test_size, random_state=random_state)
        inlier_indices = np.where(y_train == 0)[0]
        X_train = X_train[inlier_indices]
        y_train = y_train[inlier_indices]
        return X_train, X_test, y_train, y_test

def check_datasets():
   # names = ["mnist", "thyroid", "campaign", "arrhythmia", "pima", "letter", "glass","ionosphere"]
    names = ['fraud_noamount', "ionosphere", "letter", "lympho", "pima", "thyroid", "vowels", "wbc","vertebral", "arrhythmia", "musk", "satellite"]
    infdf=pd.DataFrame({'Name':[],
                        'num_samples':[],
                        'dim':[],
                        'rate':[]
                        })
    for name in names:
        Dataset = DATASET(name=name, scalertype=None)
        infdf = infdf.append({'Name': Dataset.name,
                              'num_samples': int(Dataset.numSample),
                              'dim': int(Dataset.dim),
                              'rate': Dataset.rate
                              }, ignore_index=True)
    infdf.to_csv('dataset_infomation.csv')

    infTable = tabulate(infdf, headers='keys', tablefmt='psql', showindex="never")
    print(infTable)


if __name__ == "__main__":
    # Dataset = DATASET(name="campaign", scalertype=None)
    # check_datasets()
    Dataset = DATASET(name="arrhythmia", scalertype='StandarScaler')

    #weakly-supervised (anomaly-free) 
    X_train, X_test, y_train, y_test = Dataset.weaksupervised_split(7)   

    
    
    
    
    
    