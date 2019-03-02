import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import yaml

best = {
    'cnn' : {'C': [0.1], 'gamma':[0.0001]},
    'surf': {'C': [10], 'gamma':[0.0001]},
    'vlad': {'C': [1], 'gamma':[0.0001]}
}

class SVM():

    def __init__(self, feature_file, train_list, val_list, test_list, feature_type, use_scale):
        self.lable_dict = {'P001': 1, 'P002': 2, 'P003': 3, 'NULL': 0}
        self.model = []
        self.model_a = []
        self.feature_type = feature_type
        self.scale = StandardScaler()
        self.use_scale = use_scale

        with open(feature_file, 'rb') as f:
            self.feature_dict = pickle.load(f)

        train_X = []
        train_y = []
        with open(train_list, 'r') as f:
            for line in f.readlines():
                file = line.replace('\n', '').split(' ')[0]
                label = self.lable_dict[line.replace('\n', '').split(' ')[1]]
                train_X.append(self.feature_dict[file])
                train_y.append(label)
        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)

        val_X = []
        val_y = []
        with open(val_list, 'r') as f:
            for line in f.readlines():
                file = line.replace('\n', '').split(' ')[0]
                label = self.lable_dict[line.replace('\n', '').split(' ')[1]]
                val_X.append(self.feature_dict[file])
                val_y.append(label)
        self.val_X = np.array(val_X)
        self.val_y = np.array(val_y)

        test_X = []
        test_name = []
        with open(test_list, 'r') as f:
            for line in f.readlines():
                file = line.replace('\n', '').split(' ')[0]
                test_name.append(file)
                test_X.append(self.feature_dict[file])
        self.test_X = np.array(test_X)
        self.test_name = test_name


    def train(self, g):
        self.model = []
        X = self.train_X.copy()
        if self.use_scale:
            self.scale.fit(X)
            X = self.scale.transform(X)
        for i in range(3):
            y = self.train_y.copy()
            y[y!=i+1]=0
            y[y!=0]=1
            clf = SVC()
            clf.set_params(**g)
            self.model.append(clf.fit(X, y))


    def train_all(self, g):
        X = np.concatenate([self.train_X, self.val_X], axis=0)
        if self.use_scale:
            self.scale.fit(X)
            X = self.scale.transform(X)
        for i in range(3):
            y = np.concatenate([self.train_y, self.val_y], axis=0)
            y[y!=i+1]=0
            y[y!=0]=1
            clf = SVC()
            clf.set_params(**g)
            self.model_a.append(clf.fit(X, y))


    def val(self, is_test):
        if self.model is None:
            raise Exception('Model has not been initialized')
        index = 1
        AP = []
        val_X = self.val_X.copy()
        if self.use_scale:
            val_X = self.scale.transform(val_X)
        models = self.model_a if is_test else self.model
        for model in models:
            y_true = self.val_y.copy()
            y_true[y_true!=index]=0
            y_true[y_true!=0]=1
            y_pred = model.decision_function(val_X)
            AP.append(average_precision_score(y_true, y_pred))
            index += 1
        mAP = np.mean(np.array(AP))
        return AP, mAP

    def test(self):
        if self.model is None:
            raise Exception('Model has not been initialized')
        index = 1
        test_X = self.test_X.copy()
        if self.use_scale:
            test_X = self.scale.transform(test_X)
        print('here')
        for model_a in self.model_a:
            y_pred = model_a.decision_function(test_X)
            output_file = 'SVM_output/' + 'P00' + str(index) + '_' + self.feature_type + '.lst'
            print(output_file)
            with open(output_file, 'w') as f:
                for file, score in zip(self.test_name, y_pred):
                    value = file + ' ' + str(score)
                    f.write("%s\n" % score)
            index += 1

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} feature_type config_file use_scale".format(sys.argv[0]))
        print ("feature_type -- type of feature: surf / cnn")
        print ("config_file -- yaml filepath containing all parameters")
        print ("use_scale -- if use scale in training")
        exit(1)

    f_type = sys.argv[1]
    config_file = sys.argv[2]
    use_scale = str2bool(sys.argv[3])
    my_params = yaml.load(open(config_file))

    if f_type == 'surf':
        feature_file = my_params.get('surf_his_feature')
    elif f_type == 'cnn':
        feature_file = my_params.get('cnn_his_feature')
    else:
        feature_file = my_params.get('vlad_feature')
    train_list = my_params.get('train')
    val_list = my_params.get('val')
    test_list = my_params.get('test')

    model = SVM(feature_file, train_list, val_list, test_list, f_type, use_scale)
    #grid = {'C': [0.1, 1, 10, 100], 'gamma':[0.01, 0.001, 0.0001, 0.00001, 0.000001]}
    grid = best[f_type]
    for g in ParameterGrid(grid):
        model.train_all(g)
        print(model.val(True), g)
        model.test()
