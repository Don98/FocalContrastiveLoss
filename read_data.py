import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.view_generator import ContrastiveLearningViewGenerator
from retinanet.dataloader import CocoDataset, CSVDataset, collater_train,collater_test, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer,UnNormalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
assert torch.__version__.split('.')[0] == '1'
import time
from retinanet import losses
print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--name', help='is the name')
    parser.add_argument('--n_views', help='the number to contrasive',type=int,default=2)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        data_transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=data_transform)
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater_train, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater_test, batch_sampler=sampler_val)


    print('Num training images: {}'.format(len(dataset_train)))
    the_length = len(dataset_train)
    contrastiveloss = losses.ContrastiveLoss(batch_size = 2)
    for iter_num, data in enumerate(dataloader_train):
        if(iter_num == 1):
            break
        # print(data["img"].shape,data["annot"].shape)
        features = [data["img"] for i in range(5)]
        loss = contrastiveloss(features)
        
if __name__ == '__main__':
    main()
