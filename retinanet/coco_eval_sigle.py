from pycocotools.cocoeval import COCOeval
import json
import torch

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
import time
import cv2
import os

def evaluate_coco(dataloader_val, retinanet, dataset_val,threshold=0.05):
    
    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        if(idx == 5):
            break
        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu()>0.5)
            # print(data["annot"])
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            num = len(os.listdir("result/")) // 2
            cv2.imwrite("result/" + str(num) + "_o.jpg",img)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)
            for j in data["annot"][0].numpy():
                # print(j)
                x1 = int(j[0])
                y1 = int(j[1])
                x2 = int(j[2])
                y2 = int(j[3])
                label_name = dataset_val.labels[int(j[4])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            cv2.imwrite("result/"+str(num) + "img.jpg",img)
            # cv2.imwrite("img1.jpg",img)
