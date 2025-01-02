# data-augmentation-segmentation
The Impact of Data Augmentations on Coastal Image Segmentation Models

This repository contains the code required to reproduce the results in the conference paper:

> To be updated

This code is only for academic and research purposes. Please cite the above paper if you intend to use whole/part of the code. 

## Data Files

We have used the following dataset in our analysis: 

1. The Landsat Irish Coastal Segmentation (LICS) Dataset found [here](https://zenodo.org/records/8414665).

 The data is available under the Creative Commons Attribution 4.0 International license.

## Code Files
You can find the following files in the src folder:

- `evaluate_models_lics.ipynb` Used to obtain model predictions and display all results and visualisations in the paper
- `augment_data.py` split data and apply all augmentations
- `train.py` train and fine-tune all U-Net models
- `network.py` Contains the U-Net architecture used to load the LICS model
- `evaluation.py` helper functions used to calculate all evaluation metrics
- `utils.py` additional helper functions for the project

## Additional Files

- `test_augmentions.ipynb` used to sense check that augmentations are working correctly


# Steps for running the experiments 

## 1 Format data 

Split the finetune dataset into a training and validation set:

> python src/format_data.py --input_dir /home/directory/data_augmentation/finetune --output_dir /home/directory/data_augmentation/finetune_split

## 2 Apply augmentations
The output directory should be the base directory for all the datasets (i.e. the one that contains the finetune_split
> python src/augment_data.py --input_dir /home/directory/data_augmentation/finetune_split --output_dir /home/directory/data_augmentation/


## 3 train model
Train model with appropriate parameters. For example:
> python src/train.py --model_name "TEST" --satellite landsat --early_stopping 10 --train_path /<home>/<directory>/data_augmentation/combine --device mps
