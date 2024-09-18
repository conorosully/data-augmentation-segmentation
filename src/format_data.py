# Format data
# Conor O'Sullivan
# 17/08/2024

"""This file is used to prepare the fine-tuning dataset for data augementaion"""

import numpy as np
import os
import glob
import argparse

def random_split(input_dir,p=0.9):
    """
    Randomly split the dataset into training and validation sets.
    """
    # Get the list of image files
    image_paths = glob.glob(os.path.join(input_dir, '*.npy'))
    np.random.shuffle(image_paths)

    # Split the files into training and validation sets
    split_index = int(p * len(image_paths))
    train_paths = image_paths[:split_index]
    val_paths = image_paths[split_index:]

    return train_paths, val_paths

def save_files(paths, output_dir):
    """
    Save the list of npy files to the output directory.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path in paths:
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, np.load(path))

def main():
    parser = argparse.ArgumentParser(description="Process fine-tuning dataset.")
    parser.add_argument("--input_dir", help="Directory containing image files.")
    parser.add_argument("--output_dir", default=None, help="Directory where processed files will be saved.")
    parser.add_argument("--p", default=0.9, help="Proportion of training data.")
    
    args = parser.parse_args()
  
    if args.output_dir is None:
        # Output formated to images to the same directory as the input images
        args.output_dir = args.input_dir

    # Split the dataset into training and validation sets
    train_paths, val_paths = random_split(args.input_dir, args.p)
    save_files(train_paths, os.path.join(args.output_dir, 'train'))
    save_files(val_paths, os.path.join(args.output_dir, 'val'))


if __name__ == "__main__":
    main()
