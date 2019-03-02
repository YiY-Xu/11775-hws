import numpy as np
import pickle
import sys
import yaml
from tqdm import tqdm
from numpy import linalg as LA
import multiprocessing as mp
import time
from contextlib import closing

def calculate_vlad_diff(parameters):
    name = parameters[0]
    ext = parameters[1]
    video_file = source_path + '/' + name + ext
    array = np.genfromtxt(video_file, delimiter=';')
    model = parameters[2]
    centers = model.cluster_centers_
    cluster = model.predict(array)
    residual = np.zeros(centers.shape)

    for i in range(centers.shape[0]):
        local_descriptor_indices = np.where(cluster==i)
        local_descriptor = array[local_descriptor_indices, :].reshape(-1, array.shape[1])
        r = local_descriptor - centers[i]
        residual[i] += np.sum(r, axis=0)
    residual = np.array(residual).flatten()
    residual = residual / LA.norm(residual)
    output = 'vlad/' + name
    np.save(output, residual)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} file_list select_ratio output_file".format(sys.argv[0]))
        print ("file_list -- video file list")
        print ("config_file -- yaml filepath containing all parameters")
        print ("feature_type -- type of the feature")
        exit(1)

    video_list = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))
    f_type = sys.argv[3]

    clusters = int(my_params.get('kmeans_cluster_num'))
    vlad_feature = {}

    if f_type == 'surf':
        source_path = my_params.get('surf_folder')
        ext = '.surf'
    else:
        source_path = my_params.get('cnn_folder')
        ext = '.cnn'

    with open(f_type + '_' + str(clusters) + '.model', 'rb') as f:
        model = pickle.load(f)
    # try:
    #     with open(video_list, 'r') as f:
    #         for line in tqdm(f.readlines()):
    #             file_name = line[:-1]
    #             video_file = source_path + '/' + file_name + ext
    #             array = np.genfromtxt(video_file, delimiter=';')
    #             centers = model.cluster_centers_
    #             cluster = model.predict(array)
    #             residual = np.zeros(centers.shape)

    #             for i in range(centers.shape[0]):
    #                 local_descriptor_indices = np.where(cluster==i)
    #                 local_descriptor = array[local_descriptor_indices, :].reshape(-1, array.shape[1])
    #                 r = local_descriptor - centers[i]
    #                 residual[i] += np.sum(r, axis=0)
    #             residual = np.array(residual).flatten()
    #             residual = residual / LA.norm(residual)
    #             vlad_feature[file_name] = residual
    # except Exception as e:
    #     print(e)
    # with open(my_params.get('vlad_feature'), 'wb') as f:
    #     pickle.dump(vlad_feature, f)

    ### for parallel
    parallel_parameters = []
    process_lst = []
    P = mp.Pool()

    fread = open(video_list, "r")
    
    start = time.time()

    for line in fread.readlines():
        parallel_parameters.append([line[:-1], ext, model])

    with closing(P) as p:
        P.map(calculate_vlad_diff, parallel_parameters, 100)
            
