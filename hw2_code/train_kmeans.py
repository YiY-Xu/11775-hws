#!/bin/python 

import numpy as np
import os
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import MiniBatchKMeans 
import pickle
import sys
import pandas as pd
import time
import yaml

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("config_file -- yaml filepath containing all parameters")
        print("feature_type -- type of the feature")
        exit(1)

    config_file = sys.argv[1]
    my_params = yaml.load(open(config_file))
    f_type = sys.argv[2]
    if f_type == 'surf':
        kmean_feat_file = my_params.get('surf_keam_feat_file')
        prefix = 'surf_'
    else: 
        kmean_feat_file = my_params.get('cnn_keam_feat_file')
        prefix = 'cnn_'
    
    cluster_num = int(my_params.get('kmeans_cluster_num'))
    output_file = prefix + str(cluster_num) + '.model'
    
    X = np.array(pd.read_csv(kmean_feat_file, header=None, delimiter=";").fillna(0))
    
    start = time.time()
    model = MiniBatchKMeans(n_clusters=cluster_num, batch_size=200).fit(X)
    
    print("model fit completed with " + str(time.time() - start))
    with open(output_file,'wb') as fp:
        pickle.dump(model, fp)
    #except Exception as e:
    #    print(e)

    print("K-means trained successfully!")
