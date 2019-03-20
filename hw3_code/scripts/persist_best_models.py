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

feature_types = ['l_surf', 'l_cnn', 'l_asr', 'l_mfcc', 'l_soundnet']
hyper_param = {
    'l_surf': {'C': [100], 'gamma':[0.00001], 'kernel':['rbf']},
    'l_cnn': {'C': [1], 'gamma':[0.000001], 'kernel':['rbf']},
    'l_asr': {'C': [100], 'gamma':[0.00001], 'kernel':[laplacian_kernel]},
    'l_mfcc': {'C': [100], 'gamma':[0.001], 'kernel':[laplacian_kernel]},
    'l_soundnet': {'C': [1000], 'gamma':[0.00001], 'kernel':[laplacian_kernel]}
}

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Usage: {0} config_file".format(sys.argv[0]))
        print ("config_file -- yaml filepath containing all parameters")
        exit(1)
    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))
    use_balanced_data = str2bool(my_params.get('balanced'))
    factor = float(my_params.get('factor'))
    train_list = my_params.get('train')
    val_list = my_params.get('val')
    test_list = my_params.get('test')
    later_fusion_repeat = int(my_params.get('later_fusion_repeat'))

    for i in range(later_fusion_repeat):
        for f_t in feature_types:
            print(f_t)
            feature_type, feature_dim, feature_file = my_params.get(f_t).split(',')
            feature_file = '../../features/' + feature_file
            feature = Features([feature_type], [int(feature_dim)], [feature_file])
            feature.load_train(train_list)
            feature.load_val(val_list)
            feature.load_test(test_list)
            feature.get_balanced_data(factor)
            clf = SVC()
            grid = hyper_param[f_t]
            model = Model_Train(feature, clf)
            for g in ParameterGrid(grid):
                model.train(g)
                print(model.validate(), g)
            model.train_cv(grid, use_balanced_data)
            model.test()
            model.process_train_val()
            with open('../best/'+f_t+'_best_' + str(i) + '.pkl', 'wb') as f:
                pickle.dump(model, f)

