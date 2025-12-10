# Spoken-Digits classification
In this project, we include the dataset and dataloader for training an SNN using Rockpool to classify spoken digits

## Python version and repository dependencies
Python 3.10.8

### PYTORCH INSTALLATION
Install the version that fits best your set up (your nvcc version if using GPU or CPU)
https://pytorch.org/get-started/previous-versions/

## Dataset
This repository uses this dataset:
https://zenkelab.org/datasets/hd_audio.tar.gz

Once you download the data you can run the datamodule.py script.
The script will start cacheing the data you will need for training.

## Training a model
In the notebook spoken_digits_train_tutorial.ipynb you can find more information on how to train 
an SNN with Rockpool.
