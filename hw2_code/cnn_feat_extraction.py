from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import os
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import yaml
import numpy as np
import inceptionresnet
import vgg

class Imgdataset(Dataset):
    def __init__(self, file_list, transform, img_path, img_dim):
        self.img_path = img_path
        self.img_dim = img_dim
        self.transform = transform
        self.list = []
        with open(file_list, 'r') as f:
            for line in f.readlines():
                self.list.append(line[:-1])

    def __getitem__(self, index):
        base_name = self.list[index]
        feature = np.empty((0, 3, self.img_dim, self.img_dim))
        for i in range(20):
            filename = self.img_path + '/' + base_name + '_' + str(i) + '.jpg' #self.img_path + '/' + 
            print(filename)
            if os.path.exists(filename) == False:
                break
            im = Image.open(filename)
            img = self.transform(im)
            img = img.view(1, 3, self.img_dim, self.img_dim)
            feature = np.concatenate((feature, img), axis=0)
        return feature, base_name

    def __len__(self):
        return (len(self.list))

def loadpretrain(cnn_type):
    if cnn_type == 'resnet':
        return inceptionresnet.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    elif cnn_type == 'vgg':
        return vgg.VGG(vgg.make_layers(vgg.cfg['E'], batch_norm=True))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} video_list config_file cnn_type".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        print("cnn_type -- resnet or vgg")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    cnn_type = sys.argv[3]
    my_params = yaml.load(open(config_file))

    captures_path = my_params.get('captures_folder')
    output_path = my_params.get('img_features')
    batch_size = int(my_params.get('batch_size'))
    img_dim = int(my_params.get('img_dim'))

    transform = transforms.Compose([
        transforms.Resize((img_dim)),
        transforms.CenterCrop(img_dim),
        transforms.ToTensor(),
        ])

    model = loadpretrain()
    print('CNN model is loaded')
    model = model.double()
    if torch.cuda.is_available():
        model.cuda()

    dataloader = DataLoader(
        Imgdataset(all_video_names, transform, captures_path, img_dim), batch_size=1) 
        
    model.eval()
    for i, (data, name) in enumerate(dataloader):
        data = torch.squeeze(data)
        if torch.cuda.is_available():
            data = data.cuda()
        vector = model(Variable(data))
        vector = vector.cpu().detach().numpy()

        output = output_path + '/' + name[0] + '.cnn'

        with open(output, 'w') as f:
            np.savetxt(f,vector,delimiter=';')

