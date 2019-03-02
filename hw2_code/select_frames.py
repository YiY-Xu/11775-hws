#!/bin/python
# Randomly select 
import yaml
import numpy as np
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} file_list select_ratio output_file".format(sys.argv[0]))
        print ("file_list -- the list of video names")
        print("config_file -- yaml filepath containing all parameters")
        print("feature_type -- type of the feature")
        exit(1)

    file_list = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))
    f_type = sys.argv[3]

    if f_type == 'surf':
        ratio = float(my_params.get('surf_select_frame_rate'))
        pos_ratio = ratio + 0.01
        output_file = my_params.get('surf_keam_feat_file')
        base_path = 'surf/'
        end_cut = -6
        ext = '.surf'
    else:
        ratio = float(my_params.get('cnn_select_frame_rate'))
        output_file = my_params.get('cnn_keam_feat_file')
        base_path = 'cnn/'
        pos_ratio = ratio
        end_cut = -5
        ext = '.cnn'
    positive_list = my_params.get('positive_file')

    fread = open(file_list,"r")
    fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    positive = []
    with open(positive_list, 'r') as f:
        for line in f.readlines():
            positive.append(line[:-1])

    np.random.seed(18877)

    index = 1

    for line in fread.readlines():
        try:
            if index % 10 == 0 :
                print(str(index) + ' files completed')
            filepath = base_path + line.replace('\n','') + ext
            print(filepath)
            if os.path.exists(filepath) == False:
                continue
            array = np.genfromtxt(filepath, delimiter=";")
            np.random.shuffle(array)
            
            if line[:end_cut] in positive:
                select_size = int(array.shape[0] * pos_ratio)
            else:
                select_size = int(array.shape[0] * ratio)
            if select_size == 0 :
                select_size = array.shape[0]

            if select_size > 800 and line[:-6] not in positive:
                select_size = 800

            print(select_size)
            
            feat_dim = array.shape[1]

            for n in range(select_size):
                line = str(np.float32(array[n][0]))
                for m in range(1, feat_dim):
                    line += ';' + str(array[n][m])
                fwrite.write(line + '\n')
        except Exception as e:
            print(e)
        index += 1
    fwrite.close()

