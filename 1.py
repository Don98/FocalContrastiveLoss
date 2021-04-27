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

import skimage.io
import skimage.transform
import skimage.color
import skimage

import cv2
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
def draw_pic(img,anntos,num):
    unnormalize = UnNormalizer()
    img = np.array(255 * unnormalize(img)).copy()
    transformed_anchors = anntos
    img[img<0] = 0
    img[img>255] = 255

    img = np.transpose(img, (1, 2, 0))

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    for j in range(transformed_anchors.shape[0]):
        bbox = transformed_anchors[j, :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        if(int(bbox[4]) == -1):
            continue
        # label_name = dataset_val.labels[int(bbox[4])]
        draw_caption(img, (x1, y1, x2, y2), "name")

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    cv2.imwrite("val/"+str(num) + "img.jpg",img)

def change(x1,x2,y1,y2,num,x,y):
    xx1 = int((x1 + x2) * num / 2 - x / 2)
    xx2 = xx1 + x
    yy1 = int((y1 + y2) * num / 2 - y / 2)
    yy2 = yy1 + y
    return xx1,xx2,yy1,yy2

def main(args=None):
    unnormalize = UnNormalizer()
    def to_img(the_img):
        img = np.array(255 * unnormalize(the_img)).copy()
        img[img<0] = 0
        img[img>255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return img
    img0 = torch.load("img0.pt")
    anntos0 = torch.load("anntos0.pt")
    img1 = torch.load("img1.pt")
    anntos1 = torch.load("anntos1.pt")
    
    num = 2.66
    print(img0.shape)
    # a = transforms.ToPILImage()(img0)
    # b = transforms.Resize((int(a.size[1]*num),int(a.size[0]*num)))(a)
    # b = transforms.ToTensor()(b)
    
    # a = to_img(img0)
    # print(a.shape)
    # c = transforms.ToTensor()(a)
    # b = cv2.resize(a, (int(a.shape[1]*num),int(a.shape[0]*num)), interpolation=cv2.INTER_CUBIC)
    # print(b.shape)
    # b = transforms.ToTensor()(b)
    img1 = img0.permute(1,2,0).numpy()
    
    b = skimage.transform.resize(img1, (int(round(img0.shape[1]*num)), int(round((img0.shape[2]*num)))))
    b = torch.from_numpy(b).permute(2,0,1)
    print(img0.shape,b.shape)
    
    draw_pic(img0,anntos0,"0")
    for i in anntos0:
        x1 , x2 , y1, y2 = int(i[0]),int(i[2]),int(i[1]),int(i[3])
        x = x2 - x1
        y = y2 - y1
        xx1 ,xx2,yy1,yy2 = change(x1,x2,y1,y2,num,x,y)
        img0[:,y1:y2,x1:x2] = b[:,yy1:yy2,xx1:xx2]
    draw_pic(img0,anntos0,"1")
    img0 = to_img(img0)
    b = to_img(b)
    cv2.imwrite("val/img0.jpg",img0)
    cv2.imwrite("val/img1.jpg",b)
    
if __name__ == '__main__':
    main()
