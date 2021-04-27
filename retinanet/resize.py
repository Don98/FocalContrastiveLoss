from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import skimage.io
import skimage.transform
import skimage.color
import skimage
def change(x1,x2,y1,y2,num,x,y):
    xx1 = int((x1 + x2) * num / 2 - x / 2)
    xx2 = xx1 + x
    yy1 = int((y1 + y2) * num / 2 - y / 2)
    yy2 = yy1 + y
    return xx1,xx2,yy1,yy2
    
def to_resize(sample, num,n_views):
    image, annots = sample['img'], sample['annot']
    # print("The image shape is ",image.shape,annots.shape)
    imgs = [resize_GT(image[i].clone(),annots[i].clone(),num) for i in range(image.shape[0])]
    length = sample['img'].shape[0]
    nimgs = torch.zeros((length * n_views ,sample['img'].shape[1],sample['img'].shape[2],sample['img'].shape[3]))
    nannots = torch.zeros((length * n_views ,sample['annot'].shape[1],sample['annot'].shape[2]))
    for i in range(length):
        nimgs[i*2]   = image[i]
        nimgs[i*2+1] = imgs[i]
        nannots[i*2] = annots[i]
        nannots[i*2+1] = annots[i]
    sample['img'] = nimgs
    sample['annot'] = nannots
    return sample

def resize_GT(img,annot,num):
    img1 = img.permute(1,2,0).numpy()
    b = skimage.transform.resize(img1, (int(round(img.shape[1]*num)), int(round((img.shape[2]*num)))))
    b = torch.from_numpy(b).permute(2,0,1)
    for i in annot:
        x1 , x2 , y1, y2 = int(i[0]),int(i[2]),int(i[1]),int(i[3])
        x = x2 - x1
        y = y2 - y1
        xx1 ,xx2,yy1,yy2 = change(x1,x2,y1,y2,num,x,y)
        img[:,y1:y2,x1:x2] = b[:,yy1:yy2,xx1:xx2]
    return img
