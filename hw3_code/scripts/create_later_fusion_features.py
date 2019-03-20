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

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Usage: {0} config_file".format(sys.argv[0]))
        print ("config_file -- yaml filepath containing all parameters")
        exit(1)
    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))
    train_list = my_params.get('train')
    val_list = my_params.get('val')
    test_list = my_params.get('test')
    later_fusion_repeat = int(my_params.get('later_fusion_repeat'))
    later_fusion_features = {}
    for j in range(later_fusion_repeat):
        late_fusion = np.zeros((2935, 1))

        for f_t in feature_types:
            with open('../best/' + f_t + '_best_' + str(j) + '.pkl', 'rb') as f:
                model = pickle.load(f)
                for i in range(3):
                    vector = np.concatenate((model.train_result[i].reshape(-1, 1), model.val_result[i].reshape(-1, 1)), axis=0)
                    vector = np.concatenate((vector, model.test_result[i].reshape(-1, 1)), axis=0)
                    late_fusion = np.concatenate((late_fusion, vector), axis=1)

        late_fusion = late_fusion[:, 1:]
        
        files = [train_list, val_list, test_list]
        base = 0
        tracker = 0
        for lst in files:
            with open(lst, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    name = line.replace('\n','').split(' ')[0]
                    if name in later_fusion_features.keys():
                        later_fusion_features[name] = np.concatenate((later_fusion_features[name], late_fusion[i + base].reshape(1, -1)), axis=1)
                    else:
                        later_fusion_features[name] = late_fusion[i + base].reshape(1, -1)
                    tracker = i+base
            base = tracker+1

    with open('../../features/later_fusion_features.pkl', 'wb') as f:
        pickle.dump(later_fusion_features, f)



    