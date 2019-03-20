#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores, d) computation of class labels for kaggle submission.

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -k true -y filepath

# Reading of all arguments:
while getopts e:l:d:k: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	e) EARLY_F=${OPTARG};;             # boolean true or false
	l) LATE_F=${OPTARG};;              # boolean
    d) DOUBLE_F=${OPTARG};;            # boolean
	k) KAGGLE=$OPTARG;;                # boolean
	esac
	done

export PATH=~/anaconda3/bin:$PATH

clusters=2000

cd scripts

if [ "$EARLY_F" = true ] ; then

    echo "#####################################"
    echo "#         Early Fusion              #"
    echo "#####################################"

    python early_fusion ../config.yaml

fi

if [ "$LATE_F" = true ] ; then

    echo "#####################################"
    echo "#  Persist Best Models              #"
    echo "#####################################"

    python persist_best_models.py ../config.yaml


	echo "#####################################"
    echo "#   Create Late Fusion Features     #"
    echo "#####################################"

    python create_later_fusion_features.py ../config.yaml

    echo "#####################################"
    echo "#   Late Fusion Classifier      #"
    echo "#####################################"

    python later_fusion.py ../config.yaml

    python create_submission.py LF False

    python evaluate.py
fi

if [ "$DOUBLE_F" = true ] ; then

    echo "#####################################"
    echo "#  Persist Best Models              #"
    echo "#####################################"

    python persist_best_models.py ../config.yaml


    echo "#####################################"
    echo "#   Create Late Fusion Features     #"
    echo "#####################################"

    python create_later_fusion_features.py ../config.yaml

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    python double_fusion.py ../config.yaml
fi


if [ "$KAGGLE" = true ] ; then

    echo "#####################################"
    echo "#  Persist Best Models              #"
    echo "#####################################"

    python persist_best_models.py ../config.yaml

    echo "#####################################"
    echo "#   Create Late Fusion Features     #"
    echo "#####################################"

    python create_later_fusion_features.py ../config.yaml

    echo "##########################################"
    echo "# For Kaggle Competition                 #"
    echo "##########################################"

    python kaggle.py ../config.yaml

    echo "##########################################"
    echo "# Create submission.csv                  #"
    echo "##########################################"

    python create_submission.py K True
    
    echo "##########################################"
    echo "# Evaluate the results                   #"
    echo "##########################################"
    
    python evaluate.py
fi
