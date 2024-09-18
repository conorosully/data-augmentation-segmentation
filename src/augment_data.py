# Format data
# Conor O'Sullivan
# 17/08/2024

"""This file is used to augment the dataset for training the segmentation model
Note: all these augmentations should be applied to scaled images (0-1)"""

import numpy as np
import os
import glob
import argparse
import tqdm
import utils

def rotate(image,angle=0):
    """
    Apply rotation to the npy file.
    """

    if angle == 0:
        return image
    
    return np.rot90(image, k=angle//90)

def flip(image, axis=0):
    """
    Apply flip to the npy file.

    """

    if axis == None:
        return image
    
    return np.flip(image, axis=axis)

def add_normal_noise(image,std=0.1):
    """
    Add random noise to the npy file.
    """

    if std == 0:
        return image

    # Only add noise to spectral bands
    image_bands = image[:,:,0:7]
    noise = np.random.normal(0, std, image_bands.shape)
    image_bands = image_bands + noise
    image_bands = np.clip(image_bands, 0, 1)

    return np.dstack((image_bands,image[:,:,7:]))

def add_sp_noise(image,p=0.1):
    """
    Add salt and pepper noise to the npy file.
    """

    if p == 0:
        return image

    min = 0 
    max = 1
    image_bands = image[:,:,0:7] # Only add noise to spectral bands

    # Randomly replace pixels with min or max value with probability p
    noise = np.random.choice([min, 1,max], size=image_bands.shape, p=[p/2,1-p,p/2])
    image_bands = image_bands * noise
    image_bands = np.clip(image_bands, 0, 1)

    return np.dstack((image_bands,image[:,:,7:]))

def add_contrast(image,contrast_factor=0.4):
    """
    Apply contrast adjustment to a multi-spectral image.
    """

    if contrast_factor == 0:
        return image

    image_bands = image[:,:,0:7]
    img_contrast = image_bands*contrast_factor
    img_contrast = np.clip(img_contrast, 0, 1)
    #img_contrast = cv2.convertScaleAbs(image_bands, alpha=1 + contrast_factor, beta=0)  # alpha is contrast adjustment

    return np.dstack((img_contrast,image[:,:,7:]))

def add_brightness(image,brightness_factor=0.4):
    """
    Apply brightness adjustment to a multi-spectral image.
    """

    if brightness_factor == 0:
        return image

    image_bands = image[:,:,0:7]
    img_brightness = image_bands + brightness_factor
    img_brightness = np.clip(img_brightness, 0, 1)
    #img_brightness = cv2.convertScaleAbs(image_bands, alpha=1, beta=brightness_factor)  # beta is brightness adjustment

    return np.dstack((img_brightness,image[:,:,7:]))

def aug_geometry(paths, output_dir):
    """
    Apply geometric augmentations to the npy file.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path in tqdm.tqdm(paths): 
        image = np.load(path)
        # Scale the bands
        bands = image[:,:,0:7]
        scaled_bands = utils.scale_bands(bands,satellite='landsat')
        image = np.concatenate((scaled_bands,image[:,:,7:]),axis=2)

        filename = os.path.basename(path)
        
        # Rotate the image
        for angle in [0, 90, 180, 270]:
            rotated_image = rotate(image, angle)
            output_path = os.path.join(output_dir, f'{filename}_rotated_{angle}.npy')
            np.save(output_path, rotated_image)

            for axis in [0, 1]:
                flipped_image = flip(rotated_image, axis)
                output_path = os.path.join(output_dir, f'{filename}_rotated_{angle}_flipped_{axis}.npy')
                np.save(output_path, flipped_image)


def aug_noise(paths, output_dir):
    """
    Apply noise augmentations to the npy file.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path in tqdm.tqdm(paths):
        image = np.load(path)
        # Scale the bands
        bands = image[:,:,0:7]
        scaled_bands = utils.scale_bands(bands,satellite='landsat')
        image = np.concatenate((scaled_bands,image[:,:,7:]),axis=2)

        filename = os.path.basename(path)

        # Add normal noise
        for std in [0, 0.1, 0.2, 0.3]:
            noisy_image = add_normal_noise(image, std)
            output_path = os.path.join(output_dir, f'{filename}_noisy_{std}.npy')
            np.save(output_path, noisy_image)

        # Add salt and pepper noise
        for p in [0.1, 0.2, 0.3]:
            noisy_image = add_sp_noise(image, p)
            output_path = os.path.join(output_dir, f'{filename}_sp_noisy_{p}.npy')
            np.save(output_path, noisy_image)

def aug_color(paths, output_dir):

    """
    Apply color augmentations to the npy file.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for path in tqdm.tqdm(paths):
        image = np.load(path)
        # Scale the bands
        bands = image[:,:,0:7]
        scaled_bands = utils.scale_bands(bands,satellite='landsat')
        image = np.concatenate((scaled_bands,image[:,:,7:]),axis=2)

        filename = os.path.basename(path)

        # Add contrast
        for contrast_factor in [0, 0.6, 0.8, 1.2, 1.4]:
            contrasted_image = add_contrast(image, contrast_factor)
            output_path = os.path.join(output_dir, f'{filename}_contrast_{contrast_factor}.npy')
            np.save(output_path, contrasted_image)

        # Add brightness
        for brightness_factor in [-0.3, -0.1, 0.1, 0.3]:
            bright_image = add_brightness(image, brightness_factor)
            output_path = os.path.join(output_dir, f'{filename}_brightness_{brightness_factor}.npy')
            np.save(output_path, bright_image)

def aug_combine(paths, output_dir):
    """
    Combine the augmentations to the npy file.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    for path in tqdm.tqdm(paths):
        i = 0
        image = np.load(path)

        # Scale the bands
        bands = image[:,:,0:7]
        scaled_bands = utils.scale_bands(bands,satellite='landsat')
        image = np.concatenate((scaled_bands,image[:,:,7:]),axis=2)
    
        filename = os.path.basename(path)

        # Rotate the image
        
        for angle in [0, 90, 180, 270]:
            rotated_image = rotate(image, angle)
            output_path = os.path.join(output_dir, f'{filename}_combined_{i}.npy')
            np.save(output_path, rotated_image)
            i += 1

        for axis in [0, 1]:
            flipped_image = flip(image, axis)
            output_path = os.path.join(output_dir, f'{filename}_combined_{i}.npy')
            np.save(output_path, flipped_image)
            i += 1
                
        for std in [0.1, 0.2,0.3]:
            noisy_image = add_normal_noise(image, std)
            output_path = os.path.join(output_dir, f'{filename}_combined_{i}.npy')
            np.save(output_path, noisy_image)
            i += 1

        # Add salt and pepper noise
        for p in [0.1, 0.2,0.3]:
            sp_image = add_sp_noise(image, p)
            output_path = os.path.join(output_dir, f'{filename}_combined_{i}.npy')
            np.save(output_path, sp_image)
            i += 1

        # Add contrast
        for contrast_factor in [0.6, 0.8, 1.2, 1.4]:
            contrasted_image = add_contrast(image, contrast_factor)
            output_path = os.path.join(output_dir, f'{filename}_combined_{i}.npy')
            np.save(output_path, contrasted_image)
            i += 1

        # Add brightness
        for brightness_factor in [-0.3, -0.1, 0.1, 0.3]:
            bright_image = add_brightness(image, brightness_factor)
            output_path = os.path.join(output_dir, f'{filename}_combined_{i}.npy')
            np.save(output_path, bright_image)
            i += 1

                        
def main():
    parser = argparse.ArgumentParser(description="Augment dataset.")
    parser.add_argument("--input_dir", help="Directory containing image files.")
    parser.add_argument("--output_dir", help="Directory where processed files will be saved.")

    args = parser.parse_args()

    # Get the list of image files 
    train_dir = os.path.join(args.input_dir, 'train')
    train_paths = glob.glob(os.path.join(train_dir, '*.npy'))

    val_dir = os.path.join(args.input_dir, 'val')
    val_paths = glob.glob(os.path.join(val_dir, '*.npy'))

    # Apply the specified augmentation
    output_dir = os.path.join(args.output_dir, "geometry")
    aug_geometry(train_paths, os.path.join(output_dir, 'train'))
    aug_geometry(val_paths, os.path.join(output_dir, 'val'))
    assert len(glob.glob(os.path.join(output_dir, 'train', '*.npy'))) == 1080

    output_dir = os.path.join(args.output_dir, "noise")
    aug_noise(train_paths, os.path.join(output_dir, 'train'))
    aug_noise(val_paths, os.path.join(output_dir, 'val'))
    assert len(glob.glob(os.path.join(output_dir, 'train', '*.npy'))) == 630

    output_dir = os.path.join(args.output_dir, "color")
    aug_color(train_paths, os.path.join(output_dir, 'train'))
    aug_color(val_paths, os.path.join(output_dir, 'val'))
    assert len(glob.glob(os.path.join(output_dir, 'train', '*.npy'))) == 810

    output_dir = os.path.join(args.output_dir, "combine")
    aug_combine(train_paths, os.path.join(output_dir, 'train'))
    aug_combine(val_paths, os.path.join(output_dir, 'val'))
    assert len(glob.glob(os.path.join(output_dir, 'train', '*.npy'))) == 1800


if __name__ == "__main__":
    main()
