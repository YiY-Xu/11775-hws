#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import time
import multiprocessing as mp
import subprocess
from contextlib import closing


def get_surf_features_from_video(parameters):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    # TODO
    # Get the image

    downsampled_video_filename = parameters[0]
    surf_feat_video_filename = parameters[1]
    keyframe_interval = parameters[2]
    image_feat_filename = parameters[3]

    surf_fea = np.empty((0, 128))
    index = 0

    for image in get_keyframes(downsampled_video_filename, keyframe_interval):
        image_g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        surf = cv2.SURF(500)

        keypoints, descriptors = surf.detectAndCompute(image_g, None)
        if descriptors is not None:
            image_filename = image_feat_filename + '_' + str(index) + '.jpg'
            cv2.imwrite(image_filename, image)
            index +=1
            descriptors = descriptors.reshape((-1,128))
            surf_fea = np.concatenate((surf_fea, descriptors), axis=0)
            print (surf_fea.shape, len(keypoints))

    #q.put(downsampled_video_filename + " done")
    with open(surf_feat_video_filename, 'w') as f:
        np.savetxt(f,surf_fea,delimiter=';')

    # Run the cv2 surf feature extraction


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."
    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = int(my_params.get('keyframe_interval'))
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_folder')
    downsampled_videos = my_params.get('downsampled_videos')
    image_features_folderpath = my_params.get('captures_folder')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    parallel_parameters = []
    process_lst = []
    P = mp.Pool()

    fread = open(all_video_names, "r")
    
    start = time.time()

    for line in fread.readlines():
        video_name = line.replace('\n', '')

        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.mp4.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')
        image_feat_filename = os.path.join(image_features_folderpath, video_name)

        if not os.path.isfile(downsampled_video_filename):
            continue

        parallel_parameters.append([downsampled_video_filename,
                                     surf_feat_video_filename, keyframe_interval, image_feat_filename])

    with closing(P) as p:
        P.map(get_surf_features_from_video, parallel_parameters, 100)

    print(time.time() - start)

