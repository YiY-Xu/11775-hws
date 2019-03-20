import numpy as np
import pickle
from tqdm import tqdm
import random
from sklearn.preprocessing import label_binarize


class Features:
    def __init__(self, types, dims, files):
        self.label_dict = {'P001':1, 'P002':2,'P003':3, 'NULL':0}
        self.feat_types = types
        self.feat_dims = {}
        self.features = {}
        for feat_type, feat_dim, feat_file in zip(types, dims, files):
            with open(feat_file, 'rb') as f:
                self.features[feat_type] = pickle.load(f)
                self.feat_dims[feat_type] = feat_dim

    def get_feature_label(self, data_list):
        y = []
        X = []
        with open(data_list, 'r') as f:
            for line in tqdm(f.readlines()):
                name, label = line.replace('\n', '').split(' ')
                vector = np.empty(0)
                for ft in self.feat_types:
                    concate = self.features[ft].get(name) if name in self.features[ft].keys() else np.zeros(self.feat_dims[ft])
                    vector = np.append(vector, concate)
                X.append(vector)
                y.append(self.label_dict[label])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def load_train(self, train_list):
        X, y = self.get_feature_label(train_list)
        self.train_X = X
        self.train_y = y
        self.train_yy = label_binarize(y, classes=[1, 2, 3])

    def load_val(self, val_list):
        X, y = self.get_feature_label(val_list)
        self.val_X = X
        self.val_y = y
        self.val_yy = label_binarize(y, classes=[1, 2, 3])

    def load_test(self, test_list):
        X = []
        names = []
        with open(test_list, 'r') as f:
            for line in tqdm(f.readlines()):
                name = line.replace('\n', '').split(' ')[0]
                vector = np.empty(0)
                for ft in self.feat_types:
                    concate = self.features[ft].get(name) if name in self.features[ft].keys() else np.zeros(self.feat_dims[ft])
                    vector = np.append(vector, concate)
                X.append(vector)
                names.append(name)
        self.test_X = np.array(X)
        self.test_name = names

    def get_balanced_data(self, factor):
        all_X = np.concatenate((self.train_X, self.val_X), axis=0)
        all_y = np.concatenate((self.train_y, self.val_y), axis=0)
        all_yy = np.concatenate((self.train_yy, self.val_yy), axis=0)
        positive_indice = np.where(all_y>0)[0]
        negative_indices = np.where(all_y==0)[0]

        sample_neg = random.sample(range(0, len(negative_indices)), int(factor * float(len(positive_indice))))

        select_neg = negative_indices[sample_neg]

        
        select_indices = np.concatenate((positive_indice, select_neg), axis=0)
        select_indices.sort(kind='mergesort')

        self.selected_X = all_X[select_indices]
        self.selected_y = all_y[select_indices]
        self.selected_yy = all_yy[select_indices]
        print(self.selected_X.shape)
