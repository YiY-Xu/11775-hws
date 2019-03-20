import numpy as np
import pickle
from tqdm import tqdm
from features import Features
from model_train import Model_Train
import yaml
from sklearn.svm.classes import SVC
from sklearn.model_selection import ParameterGrid
import sys
from sklearn.metrics.pairwise import laplacian_kernel, chi2_kernel

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

grid = {'C': [1], 'gamma':[0.000001], 'kernel':['rbf']}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Usage: {0} config_file".format(sys.argv[0]))
        print ("config_file -- yaml filepath containing all parameters")
        exit(1)
    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))
    feature_types = my_params.get('feature_types').split(',')
    dims = my_params.get('feature_dims').split(',')
    feature_dims = [int(x) for x in dims]
    feature_files = ['../../features/' + x for x in my_params.get('feature_files').split(',')]

    use_balanced_data = str2bool(my_params.get('balanced'))
    factor = int(my_params.get('factor'))
    train_list = my_params.get('train')
    val_list = my_params.get('val')
    test_list = my_params.get('test')

    features = Features(feature_types, feature_dims, feature_files)
    features.load_train(train_list)
    features.load_val(val_list)
    features.load_test(test_list)
    features.get_balanced_data(factor)

    SVC_model = Model_Train(features, SVC())
    for g in ParameterGrid(grid):
        SVC_model.train(g)
        print(SVC_model.validate(), g)
    SVC_model.train_cv(grid, use_balanced_data)
    SVC_model.test()
    SVC_model.persist_test_result('EF')