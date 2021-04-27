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
    
    import cv2
    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        
    def draw_pic(data,no):
        unnormalize = UnNormalizer()
        img = np.array(255 * unnormalize(data['img'][no, :, :, :])).copy()
        transformed_anchors = data["annot"][no]
        img[img<0] = 0
        img[img>255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        num = len(os.listdir("val/"))
        for j in range(transformed_anchors.shape[0]):
            bbox = transformed_anchors[j, :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            if(int(bbox[4]) == -1):
                continue
            label_name = dataset_val.labels[int(bbox[4])]
            draw_caption(img, (x1, y1, x2, y2), label_name)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print(label_name)
        cv2.imwrite("val/"+str(num) + "img.jpg",img)
    for iter_num, data in enumerate(dataloader_train):
        if(iter_num == 1):
            break
        print(data["img"].shape,data["annot"].shape)
        draw_pic(data,0)
        draw_pic(data,1)
        draw_pic(data,2)
        draw_pic(data,3)
        
if __name__ == '__main__':
    main()
