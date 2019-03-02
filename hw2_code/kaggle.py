from mlp import MLP
import numpy as np
import sys
import yaml
import pickle
import datetime
import torch
import torch.utils.data
from sklearn.neural_network import MLPClassifier

def load_training_data(feature_file, train_file_lists):
    label_dict = {'P001':1, 'P002':2, 'P003':3, 'NULL':0}
    train_X = []
    train_y = []
    feature_dict = {}
    with open(feature_file, 'rb') as f:
        feature_dict = pickle.load(f)

    for file in train_file_lists:
        print(file)
        with open(file, 'r') as f:
            for line in f.readlines():
                name = line.replace('\n', '').split(' ')[0]
                label = label_dict[line.replace('\n', '').split(' ')[1]]
                if label!=0 :
                    x = np.sum(feature_dict[name], axis=0)
                    train_X.append(x)
                    train_y.append(label)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    return train_X, train_y


def load_test_data(feature_file, test_file_lists):
    test = []
    test_filenames = []
    with open(feature_file, 'rb') as f:
        feature_dict = pickle.load(f)
    with open(test_file_lists, 'r') as f:
        for line in f.readlines():
            name = line.replace('\n', '').split(' ')[0]
            if (len(name) > 0):
                x = np.sum(feature_dict[name], axis=0)
                test.append(x)
                test_filenames.append(name)
    return np.array(test), test_filenames

def MLP_Model():
    return MLPClassifier(hidden_layer_sizes=(500, 100), activation='relu', alpha=1e-04, batch_size='auto',
              learning_rate_init=0.001, solver='adam', shuffle=True, verbose=True, early_stopping=True, n_iter_no_change=2, validation_fraction=0.05)

def model_test(model, test, test_filenames):
    test_X = test
    y_pred = model.predict(test_X)
    now = datetime.datetime.now()
    output_file = str(now.minute) +'_'+ str(now.second) + '_submission.csv'
    with open(output_file, 'w') as f:
        for file, score in zip(test_filenames, y_pred):
            value = file + ', ' + str(score)
            f.write("%s\n" % value)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Usage: {0} config file".format(sys.argv[0]))
        exit(1)

    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))
    size = my_params.get('MLP_size').split(',')
    size = [int(x) for x in size]
    feature_file = my_params.get('cnn_incept_feature')
    train_list = my_params.get('train')
    val_list = my_params.get('val')
    test_list = my_params.get('test')
    # param_file = my_params.get('MLP_param')
    # learning_rate = my_params.get('MLP_lr')
    # epochs = my_params.get('MLP_epoch')

    model = MLP_Model()
    train_X, train_y = load_training_data(feature_file, [train_list, val_list])
    model.fit(train_X, train_y)

    test, test_filenames = load_test_data(feature_file, test_list)
    model_test(model, test, test_filenames)
    


