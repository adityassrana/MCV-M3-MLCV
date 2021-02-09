# Usage python make_mini_dataset.py -s '/home/adityassrana/datatmp/Datasets/MIT_split' -t 'mini'

import argparse
import sys
from multiprocessing import Pool
from functools import partial
from path import Path
import shutil
import os
import glob
import pickle
import numpy as np

def get_folder(image_path):
    """
    returns
    """
    return os.path.join(*image_path.split('/')[-3:-1])

def copy_image(image_path, directory=str):
    """
    Copies an image to another folder keeping its
    folder structure. For eg. '/Databases/MIT_split/train/Opencountry/art582.jpg'
    with directory='mini' will create 
    
    Args:
        image_path: 
            path of image to copy
        directory: 
            directory where to copy the image along
            with its folder structure
            
    Returns:
        path to copied image
    """
    base_folder = get_folder(image_path)
    target_path = os.path.join(directory,base_folder)
    try:
        os.makedirs(target_path)
    except FileExistsError:
        pass
    return shutil.copy(image_path, target_path)

def get_image_paths_mini(image_paths, labels, samples_per_class=5):
    
    # get unique classses
    classes = np.unique(np.array(labels))
    num_classes = len(classes)
    #set size for plot
    mini_list = []
    for y, cls in enumerate(classes):
        _idxs = np.flatnonzero(np.array(labels) == cls)
        idxs = np.random.choice(_idxs, samples_per_class, replace=False)
        [mini_list.append(image_paths[idx]) for idx in idxs]
    return mini_list

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Mini DataSet Generator')

    parser.add_argument(
        "--source_directory","-s",
        help = "path to source directory. Example input: '/home/data/MIT_split")

    parser.add_argument(
        "--target_directory","-t",
        help = "path to target directory. Example input: '/home/data/mini_split")

    parser.add_argument(
        "--samples_per_class","-n", default="50", type=int,
        help = "number of samples to keep per class")

    args = parser.parse_args(args)
    return args

if __name__ == '__main__':
    
    args = parse_args()
    print(args)

    train_labels = pickle.load(open('Data/train_labels.dat','rb'))
    source_dir = Path(args.source_directory)

    train_images_filenames = sorted(glob.glob(f"{source_dir/'train/*/*.jpg'}"))
    mini_image_paths = get_image_paths_mini(train_images_filenames, train_labels, samples_per_class=args.samples_per_class)


    print(f'Sourcing data from {args.source_directory}')
    print(f'Transferring data to {args.target_directory}')

    print(f'Original Dataset Size: {len(train_images_filenames)}')
    print(f'Mini Dataset Size: {len(mini_image_paths)}')

    copy_to_dir = partial(copy_image, directory=args.target_directory)

    pool = Pool()#makes the whole process super fast
    pool.map(copy_to_dir, mini_image_paths)