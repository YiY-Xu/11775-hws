Full pipeline is given in the shell script **run.pipeline.sh**.

Arguments are passed to this bash script defining which one of the steps (preprocessing: **p**, feature representation: **f**, MAP scores: **m**, kaggle results: **k**, yaml filepath: **y**) needed to be performed.

To execute the script: 

    bash run.pipeline.sh -p true -f true -m true -k true -y