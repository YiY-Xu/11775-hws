import numpy as np
import pickle
import os
from tqdm import tqdm

if __name__=='__main__':
    vlad_features = {}
    with open('all.video', 'r') as f:
        for line in tqdm(f.readlines()):
            name = line[:-1]
            print(name)

            file = 'vlad/' + name + '.npy'
            #file = 'cnn/' + name + '.cnn'
            if not os.path.exists(file):
                continue
            #value = np.genfromtxt(file, delimiter=';')
            value = np.load(file)
            vlad_features[name] = value
            print(value.shape)

    with open('vlad_features.pkl', 'wb') as f:
        pickle.dump(vlad_features, f)
