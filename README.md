# data-augmentation-segmentation
Coastline image segmentation using data augmentation methods


# Steps for running experiment 

## 1 Format data 

Split the finetune dataset it a training and validation set. 

python src/format_data.py --input_dir /home/people/22205288/scratch/data_augmentation/finetune --output_dir /home/people/22205288/scratch/data_augmentation/finetune

## 2 Apply augmentations
The output directory should be the base directory for all the datasets (i.e. the one that contains the finetune_split
data-augmentation-segmentation]$ python src/augment_data.py --input_dir /home/people/22205288/scratch/data_augmentation/finetune_split --output_dir /home/people/22205288/scratch/data_augmentation/


## 3 train model

python src/train.py --model_name "TEST" --satellite landsat --early_stopping 10 --train_path /Users/conorosullivan/Documents/git/COASTAL_MONITORING/data-augmentation-segmentation/data/combine --device mps
