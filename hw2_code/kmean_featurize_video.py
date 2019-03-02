import numpy as np
import json
from tqdm import tqdm
import sys
import pickle
import yaml

def hist2vec(array, dim):
    vec = np.zeros((1, dim))
    unique, counts = np.unique(array, return_counts=True)
    hist = dict(zip(unique, counts))
    
    for key, value in hist.items():
        vec[0, key] = value
    return vec[0]

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        print("feature_type -- type of the feature")
        exit(1)

    video_list = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))
    f_type = sys.argv[3]

    feature_dict = {}

    if f_type == 'surf':
        source_path = my_params.get('surf_folder')
        prefix = 'surf_'
        output_path = my_params.get('surf_his_feature')
        ext = '.surf'
    else:
        source_path = my_params.get('cnn_folder')
        prefix = 'cnn_'
        output_path = my_params.get('cnn_his_feature')
        ext = '.cnn'

    cluster_num = my_params.get('kmeans_cluster_num')
    model_file = prefix + str(cluster_num) + '.model'
    

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    with open(video_list, 'r') as f:
        for line in tqdm(f.readlines()):
            video_file = source_path + '/' + line[:-1] + ext
            print(video_file)
            array = np.genfromtxt(video_file, delimiter=';')
            pred = model.predict(array)
            feature_dict[line[:-1]] = hist2vec(pred, cluster_num)

    with open(output_path, 'wb') as f:
        pickle.dump(feature_dict, f)
