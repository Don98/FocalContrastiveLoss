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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
    # Create the model
    retinanet = model.resnet18(num_classes=80, pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(118287))
    
    img = torch.load("img.pt")
    annot = torch.load("annot.pt")
    classification_loss, regression_loss, contrastiveloss = retinanet([img.cuda().float(), annot])
    print(classification_loss,regression_loss,contrastiveloss)
if __name__ == '__main__':
    main()
