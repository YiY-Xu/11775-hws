#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores, d) computation of class labels for kaggle submission.

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -k true -y filepath

# Reading of all arguments:
while getopts p:f:m:k:y: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

export PATH=~/anaconda3/bin:$PATH

clusters=2000

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=video  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans images captures vlad # create folders to save features
    awk '{print $1}' ../all_trn.lst > train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../all_val.lst > val.video
    cp ../all_test.video test.video
    cat train.video val.video test.video > all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    
    start=`date +%s`
    cd $video_path
    ls *.mp4 | parallel -I% --max-args 1 ffmpeg -y -ss 0 -i % -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate ../downsampled_videos/%.ds.mp4 #-max_muxing_queue_size 1024
    cd ..
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" 

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
    python surf_feat_extraction.py all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
	python cnn_feat_extraction.py all.video config.yaml resnet

fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: Train kmeans to obtain clusters for SURF features
    python select_frames.py all.video config.yaml surf
    python train_kmeans.py config.yaml surf


    # 2. TODO: Create kmeans representation for SURF features
    python kmean_featurize_video.py all.video config.yaml surf


	echo "#####################################"
    echo "#   CNN RESNET FEATURE REPRESENTATION      #"
    echo "#####################################"

	# 1. TODO: Train kmeans to obtain clusters for CNN features
    python select_frames.py all.video config.yaml cnn
    python train_kmeans.py config.yaml cnn

    # 2. TODO: Create kmeans representation for CNN features
    python kmean_featurize_video.py all.video config.yaml cnn

    # echo "#####################################"
    # echo "#   CNN VGG FEATURE REPRESENTATION      #"
    # echo "#####################################"

    # # 1. TODO: Train kmeans to obtain clusters for CNN features
    # python select_frames.py all.video config.yaml cnn
    # python train_kmeans.py config.yaml cnn

    # # 2. TODO: Create kmeans representation for CNN features
    # python kmean_featurize_video.py all.video config.yaml cnn

    echo "#####################################"
    echo "#   Vlad FEATURE REPRESENTATION      #"
    echo "#####################################"

    # 1. TODO: create vlad presentation
    python vlad_encoding.py all.video config.yaml surf
fi

if [ "$MAP" = true ] ; then
    mkdir -p SVM_output
    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission
    #python train_svm.py surf config.yaml true
    python train_svm.py surf config.yaml false

    echo "#######################################"
    echo "# MED with CNN RESNET Features: MAP results  #"
    echo "#######################################"


    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission
    #python train_svm.py cnn config.yaml true
    python train_svm.py cnn config.yaml false

    echo "#######################################"
    echo "# MED with Vlad Features: MAP results  #"
    echo "#######################################"


    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

    # 3. TODO: Train SVM with OVR using videos in training and validation set.

    # 4. TODO: Test SVM with test set saving scores for submission
    python train_svm.py vlad config.yaml surf
fi


if [ "$KAGGLE" = true ] ; then

    echo "##########################################"
    echo "# MED with SURF Features: KAGGLE results #"
    echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

    # 4. TODO: Test SVM with test set saving scores for submission


    echo "##########################################"
    echo "# MED with CNN Features: KAGGLE results  #"
    echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission
    python kaggle.py config.yaml

fi
