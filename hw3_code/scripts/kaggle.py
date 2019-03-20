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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

class MyModel:

    def __init__(self):
        self.grids = {
        "SVC": {'C': [1000], 'gamma':[0.000001], 'kernel':['rbf']},
        "MLP": {'hidden_layer_sizes':[(500, 100)], 'activation':['relu'], 'alpha':[1e-04], 'batch_size':['auto'],
        'learning_rate_init':[0.001], 'solver':['adam'], 'shuffle':[True], 'verbose':[True], 'early_stopping':[True], 'n_iter_no_change':[2], 'validation_fraction':[0.05]},
        'RFC': {}
        }

    def get_Model(self,type):
        if type == 'MLP':
            return self.MLP_Model()
        if type == 'SVC':
            return self.SVC_Model()
        if type == 'RFC':
            return self.RFC_Model()

    def MLP_Model(self):
        return MLPClassifier()

    def SVC_Model(self):
        return SVC()

    def RFC_Model(self):
        return RandomForestClassifier()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Usage: {0} config_file".format(sys.argv[0]))
        print ("config_file -- yaml filepath containing all parameters")
        exit(1)
    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))

    feature_types = my_params.get('k_feature_types').split(',')
    dims = my_params.get('k_feature_dims').split(',')
    feature_dims = [int(x) for x in dims]
    feature_files = ['../../features/' + x for x in my_params.get('k_feature_files').split(',')]

    models = my_params.get('k_models').replace('\n','').split(',')

    use_balanced_data = str2bool(my_params.get('balanced'))
    factor = float(my_params.get('factor'))
    train_list = my_params.get('train')
    val_list = my_params.get('val')
    test_list = my_params.get('test')
    ensemble = int(my_params.get('ensemble'))


    final = np.zeros((1699, 1))
    for i in range(ensemble):
        mymodel = MyModel()
        features = Features(feature_types, feature_dims, feature_files)
        features.load_train(train_list)
        features.load_val(val_list)
        features.load_test(test_list)
        features.get_balanced_data(factor)
        for model_name in models:
            model = Model_Train(features, mymodel.get_Model(model_name))
            model.train_cv_multiclass(mymodel.grids[model_name], use_balanced_data)
            model.test(True)

            predict = []
            result = model.test_result[0]
            for data in result:
                if data[0] == 1:
                    predict.append(1)
                elif data[1] == 1:
                    predict.append(2)
                elif data[2] == 1:
                    predict.append(3)
                else:
                    predict.append(0)
            predict = np.array(predict).reshape(-1, 1)
            final = np.concatenate((final, predict), axis=1).astype(int)
    final = final[:, 1:]

    #majority voting
    from scipy.stats import mode

    predict = list(mode(final, axis=1)[0].squeeze())

    names = []
    with open('../../all_test_fake.lst', 'r') as f:
        for line in f.readlines():
            name = line.replace('\n','').split(' ')[0]
            names.append(name)

    with open('../output/submission.csv', 'w') as f:
        f.write("VideoID,label\n")
        for name, result in zip(names, predict):
            item = name + ',' + str(result)
            f.write("%s\n" % item)

    




