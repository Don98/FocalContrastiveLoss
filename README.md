# 代码修改记录

## 2021-04-23

### 尝试完成对比学习部分。

对比学习：我采用的策略

- 正例：同一个GT中的数据增广类似于ImageNet任务中的整张图片的增广
- 负例：同一张图片图片中不是同一个类别的GT的比较，如果没有的话负例就设定为背景
  - 还有种情况下娶不负例的，比如451496、346259，只有一个检测目标，并且检测目标占据了图片的比较大的部分。

### 第一个问题：数据处理

- 无法做到只对GT进行数据增广
- 原因：在于在比对的时候才进行数据增广，这个时候图像已经提取过特征了，所以必须在特征没有提取之前做数据增广

两个方案：

- 新想法：单独训练一个ResNet，训练好之后再在RetinaNet上训练
- 原始想法：训练检测模型的同时训练ResNet，将对比学习的loss也加在模型的损失函数上面
- 谋生新想法的一个原因就是，在训练检测模型的时候，会有一个数据增广的数据是训练不到的，~~这个挺奇怪的~~，感觉有点浪费的样子，但是认真思考之后好像确实这样子才可以（毕竟是检测数据集，~~但是好像也能做到把GT拿出来当作一个新的数据集？不再对这方面进行思考~~）

在输入的时候：

- 训练阶段：一次输入的大小等于batch_size*2，一张是原本的，一张是用来做对比的。
  - 就这个说法，速度可能变慢？毕竟多了一份需要提取特征的数据。
- 测试阶段：正常

具体实现：

从文件train.py开始

```python
dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
         transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
         transform=transforms.Compose([Normalizer(), Resizer()]))
```

上面分别是train和val的数据集，val不需要管，主要是关注train的内容。

原本的train中，有Normalizer(), Augmenter(), Resizer()三种方式，分别为自己实现的标准化、数据增广（百分之五十的几率翻转）和最短边缩放。

SimCLR的对比主要来自于在裁剪的对比（不完全一致），裁剪的部分可能要在比较anchor和GT的时候才能够出现了。

在这一步需要完成的任务在于

- 有个想法，对于提取好特征的GT再进行裁剪什么的，（不太合适，毕竟在同一个图片上提取的特征）
- 一次输入两个一样的数据：



代码：

~~将原本的transforms剥离出来，然后单独实现一个得到两个输出的ContrastiveLearningViewGenerator，这个内容实现在view_generator.py之中。~~

~~实现之后：~~

```python
data_transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()]
dataset_train = CocoDataset(parser.coco_path, set_name='train2017',transform=ContrastiveLearningViewGenerator(data_transform,n_views = 2)))
```

~~需要修改retinanet/dataloader.py中关于collater的类型，这一部分是将输入在提取之前进行修改的，主要是加上padding的操作：~~

```python
def collater(data):
    print("the length of the data is ", len(data))
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
```



~~修改办法，将改为~~

直接修改collater，在输入前再copy一份，上面的data_transform保持不变，方便后续增加增广手段，删除掉view_generator.py，修改collater的思路如下：

- 在原本的函数后面进行添加
- 在原本的图像之中将GT全部取出来，然后缩放、扭曲等操作之后，裁剪回去原本的大小。
- collater改为训练用一个和测试用一个

具体方案为：

- 在训练的collater之后，创造一个2~3的随机数，这个数字将会把图片放大2~3倍数。
- 然后使用在retinanet/resize.py中的to_resize函数解决问题。

```python
def collater_train(data):
    n_views = 2
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)
    
    
    o_result = {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}
    num = torch.rand(1) + 2
    result = to_resize(o_result,float(num),n_views)
    return result
```

#### 实现步骤

然后接下来是具体的to_size的实现：

- 第一步：将其中每一个图片、标签取出来
- 第二步：将取出来的图片、标签按照给定的随机数字缩放，在缩放之后，取出和原本一样大小的框，位于新框的左下角，然后用框代替原本图像克隆的框，然后返回。（使用函数resize_GT完成）。
- 第三步：去除每一个batch的图片进行处理，直到完成为止
- 第四步：将所有的图片拼接回同一个tensor，然后返回字典

```python
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

def to_resize(sample, num,n_views):
    image, annots = sample['img'], sample['annot']
    # print("The image shape is ",image.shape,annots.shape)
    imgs = [resize_GT(image[i],annots[i],num) for i in range(image.shape[0])]
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
    img1 = transforms.ToPILImage()(img).convert('RGB')
    img1 = img1.resize((int(img1.size[0]*num),int(img1.size[1]*num)))
    img1 = transforms.ToTensor()(img1)
    for i in annot:
        x1 , x2 , y1, y2 = int(i[0]),int(i[2]),int(i[1]),int(i[3])
        x = x2 - x1
        y = y2 - y1
        xx1 ,xx2,yy1,yy2 = int(x1*num),int(x1*num)+x,int(y1*num),int(y1*num) + y
        img[:,y1:y2,x1:x2] = img1[:,yy1:yy2,xx1:xx2]
    return img
```

至此，同一张图片的对比图片完成了，

```python
for epoch_num in range(parser.epochs):
    for iter_num, data in enumerate(dataloader_train):
	    print(data["img"].shape,data["annot"].shape)
```

对于如上的输出，大小为（batch_size\*n_views, channels, width,height）和(batch_size\*n_views,anchors num, 5)



至此第一部分，对于数据的处理完成。

接下来是第二部分，将对比的loss计算出来，加入到Focal Loss中的内容了。（2021.04.25.00：31）

> 好家伙，一下午的工作给我干了两天。



### 裁剪结果

在第二步开始之前还是查看一下结果吧

![first](D:\大三下\four\val\first.bmp)

嗯嗯嗯，幸好查看了一下，发现完全认不出来了。

使用clone之后变化前就恢复正常了。

经过代码的拉伸之后发现直接出现了问题：

```python
    img0 = torch.load("img0.pt")
    anntos0 = torch.load("anntos0.pt")
    img1 = torch.load("img1.pt")
    anntos1 = torch.load("anntos1.pt")
    num = 2.66
    a = transforms.ToPILImage()(img0).convert('RGB')
    b = a.resize((int(a.size[0]*num),int(a.size[1]*num)))
    b = transforms.ToTensor()(b)
    
    unnormalize = UnNormalizer()
    def to_img(the_img):
        img = np.array(255 * unnormalize(the_img)).copy()
        # img[img<0] = 0
        # img[img>255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return img
    save_img = to_img(img0)
    save_b = to_img(b)
    cv2.imwrite("val/img0.jpg",save_img)
    cv2.imwrite("val/img1.jpg",save_b)
```

![0](D:\大三下\four\val\0.jpg)

原图变大之后的场景，看来是拉伸的函数出了问题,切确的调查之后是PIL和tensor之间的转化出了问题，去除掉超出范围的就好多了。

但还是有问题，如下原版图片：

![1](D:\大三下\four\val\1.jpg)

经过修改已经完成了

代码如下：

```python
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
    b = skimage.transform.resize(img1, (int(round(img0.shape[1]*num)), int(round((img0.shape[2]*num)))))
    b = torch.from_numpy(b).permute(2,0,1)
    for i in anntos0:
        x1 , x2 , y1, y2 = int(i[0]),int(i[2]),int(i[1]),int(i[3])
        x = x2 - x1
        y = y2 - y1
        xx1 ,xx2,yy1,yy2 = change(x1,x2,y1,y2,num,x,y)
        img0[:,y1:y2,x1:x2] = b[:,yy1:yy2,xx1:xx2]
    return img

```

![2](D:\大三下\four\val\2.jpg)

![3](D:\大三下\four\val\3.jpg)

不过**确实出现了一开始想到的问题--重叠**。

> 时间：2021-04-25

### 第二部分，计算loss

在retinanet/losses.py中实现第二个类：ContrastiveLoss，这个类有输入参数batch_size和temperature。

前一个为自己的设定，不过batch_size这个估计要在model中自己输入了。

第二个参数的话具体参考论文中对于这个参数的设定。

类的初始化是这么实现的。

```python
class ContrastiveLoss(nn.Module):
    def __init__(self,batch_size,temperature = 0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
```

然后接下来就是对于FPN五个层次进行loss的计算了，一开始本来准备将它们改造成为batch_size * len(features)来实现的，但是突然意识到一个问题，features中每一个大小都不一样，所以说最后还是要分别进行计算，想了想在forward中再写一个对于每一个层次的features进行计算的函数。

```python
def forward(self,features):
    new_index = torch.arange(0,self.batch_size) * 2
    emb_i = [feature[new_index] for feature in features]
    emb_j = [feature[new_index+1] for feature in features]
    contrastiveloss = torch.zeros(0)
    #以上先把对应的聚合0、2、4……为原图所在位置
    for i in range(len(features)):
        contrastiveloss += calc_one(emb_i[i],emb_j[i])
    return contrastiveloss
	# 然后把每一个层次的feature累加
```

接下来是对于calc_one的实现，这一部分的实现参考[文章](https://blog.csdn.net/u011984148/article/details/107754554)中对于向量化的实现。

```python
def calc_one(self,emb_i,emb_j):
    z_i = F.normalize(emb_i, dim = 1)
    z_j = F.normalize(emb_j, dim = 1)
    representations = torch.cat([z_i,z_j], dim = 0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),representations.unsqueeze(0), dim = 2)
    sim_ij = torch.diag(similarity_matrix, self.batch_size)
    sim_ji = torch.diag(similarity_matrix, -self.batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)
    nominator = torch.exp(positives / self.temperature)
    denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss_partial) / (2 * self.batch_size)
    return loss
```

突然意识到好像可以直接使用。

不过这一份代码主要是整张图片的计算，我们还需要得出对应框的loss，这一部分需要我们自己实现，要好好的实验一下，不过显卡还在被使用之中。

目前的话，实现了在整张图片上的对比损失的计算，代码如下：

```python
class ContrastiveLoss(nn.Module):
    def __init__(self,batch_size,temperature = 0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
    def calc_one(self,emb_i,emb_j):
        z_i = F.normalize(emb_i, dim = 1)
        z_j = F.normalize(emb_j, dim = 1)
        
        representations = torch.cat([z_i,z_j], dim = 0)
        representations = representations.view(representations.shape[0],-1)
        similarity_matrix = representations.mm(representations.t())
        
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.log(positives / self.temperature)
        denominator = self.negatives_mask * torch.log(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    def forward(self,features):
        new_index = torch.arange(0,self.batch_size) * 2
        emb_i = [feature[new_index] for feature in features]
        emb_j = [feature[new_index+1] for feature in features]
        contrastiveloss = torch.tensor(0.0)
        for i in range(len(features)):
            contrastiveloss += self.calc_one(emb_i[i],emb_j[i])
            print("contrastiveloss is ",contrastiveloss)
        return contrastiveloss / len(features)
```

但是我们要计算的是anchor的区别，实现思路可以从下面步骤改进：

- 将原本的calc_one改为calc_GT，也就是说输入进来的时候就已经是anchor了，那么我们不需要操心接下来的内容了、
- 那么把calc_one的内容全部改到函数calc_GT下面，然后再calc_one所作的事情就是把每一个框取出来然后丢进去计算，这一步骤可能有点慢，原因在于不同的anchor的大小不一样，不能够向量化运算。
- 这一部分在后继实现（另一方面根据不同大小的anchor可以尝试给一个不同的参数）
- 需要考虑的一个问题是，GT框的大小缩放尺寸的问题，在FPN不同的地方anchor框的大小应该不一致才是的
  - 这一部分内容可以查看原本的loss中对于anchor的定位实现。

> 时间：2021-4-26-21:25

emmmm，整了一部分calc_one的内容，但是发现了一个问题，前面的torch.mm只是将两个矩阵相乘而已，所以真实的计算出余弦相似度这一个函数还需要改变：

```python
def calc_one(self,emb_i,emb_j,annotation):
    loss = torch.tensor(0.0)
    # print(annotation.shape)
    # print(emb_i.shape,emb_j.shape)
    for batch in range(self.batch_size):
        print("The batch is ",batch)
        for i in annotation[batch,:,:]:
            print(i)
            loss += self.calc_GT(emb_i[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])],emb_j[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])])
            # print("="*50)
            return loss
```

另一个问题是，计算loss的时候需要两重循环。

> 时间：2021-04-27-01：38

这一部分搞定了，采用原本的代码来计算余弦相似度就可以了。

```python
similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
```

接下来就要解决GT在不同FPN上的大小问题了。

代码也基本完成了

- 通过观察之后FPN从最低层开始的特征层大小分别是原图像大小的8、16、32、64、128倍小，所以对于对应的GT分别除以这个数字就好了，但是容易出一些问题的，比如框缩小到了重合的地步之类的。

但是似乎还有一些问题的，下午再进行查看

> 2021-04-27 11:47

#### 具体实现

大概查出问题来了，这个在代码写出来后说明，下面是整个对比学习在文件retinanet/losses/py中：

```python

class ContrastiveLoss(nn.Module):
    def __init__(self,batch_size,n_views,temperature = 0.5):
        super().__init__()
        self.batch_size = batch_size
        self.n_view = n_views
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
    def calc_GT(self,emb_i,emb_j):
        z_i = F.normalize(emb_i, dim = 1)
        z_j = F.normalize(emb_j, dim = 1)
        
        representations = torch.cat([z_i,z_j], dim = 0)
        representations = representations.view(representations.shape[0],-1)
        # similarity_matrix = representations.mm(representations.t())
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    def calc_one(self,emb_i,emb_j,annotation):
        loss = torch.tensor(0.0)
        for batch in range(self.batch_size):
            for i in annotation[batch,:,:]:
                loss += self.calc_GT(emb_i[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])],emb_j[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])])
            loss /= annotation.shape[1]
        return loss/self.batch_size
    def forward(self,features,annotations):
        new_index = torch.arange(0,self.batch_size) * self.n_view
        emb_i = [feature[new_index] for feature in features]
        emb_j = [feature[new_index+1] for feature in features]
        length= len(features)
        annot = [annotation[new_index] for annotation in annotations]
        contrastiveloss = torch.tensor(0.0)
        for i in range(length):
            contrastiveloss += self.calc_one(emb_i[i],emb_j[i],annot[i])
        return contrastiveloss / length
```

然后是对于FPN中的处理，在文件retinanet/model.py中

```python
contrastiveannots = [(annotations[:,:,:4].clone() / i).ceil() for i in [8,16,32,64,128]]
print("contrastiveannots ",len(contrastiveannots),contrastiveannots[0].shape)
if self.training:
    return self.focalLoss(classification, regression, anchors, annotations) , self.contrastiveloss(features,annotations)
```

对于test不使用这一套。

经过测试基本上没有问题的，唯一的问题在于retinanet/losses.py的类ContrastiveLoss的forward中，对于标签的处理：

```python
annot = [annotation[new_index] for annotation in annotations]
```

对于$batch size \geq 2$ 是没有问题的，但是对于$batchsize == 1$会出现一个问题就是原本shape应该为（batch size，num，5）的annot会变成（num，5）也就是当batch size等于1的时候会忽略掉第一个维度（代码问题，懒得改了）。

另外还有一些代码上的问题，batch size的设定问题。

```python
self.contrastiveloss = losses.ContrastiveLoss(batch_size = 1,n_views = 2)
```

**在模型中对于batch_size和n_views的设定必须要在文件retinanet/model.py中手动设定。这是一个小问题了。**

> 2021-04-27 18:57

### 第三部分：其他实现

因为当前没有显卡使用，所以接下来继续完成一些代码，两个工作是可以做的：

- 第一个工作：增加数据增广的手段，这一部分要做的话主要在 DataLoader中的collate_fn函数中实现，原本在这里实现的功能是将数据copy一份作为对比
  - 具体的工作就是将GT拉伸之后之后填进去，之后就作为用来对比的图像，问题在于GT可能会进行重叠、覆盖。
  - 能做的事情就是对对比的图像进行其他的数据增广操作，比如扭曲、翻转之类的，扭曲简单，翻转还需要对于annto进行修改还有些麻烦
  
- 第二个工作：我觉得这个工作应该才是我接下来在拿到显卡之前要做的工作。

  - 数据的保存，在train1.py中创建文件f，然后将参数传入进去就可以了，唯一的问题就是写数据似乎会5hki 增加一些训练时间的样子

    - 在经过实验之后，发现写数据似乎没有增加run的时间，暂时不关注了

  - 写数据的格式如下：

  - > size contrastiveloss in P3 | size contrastiveloss in P4 | …… |size contrastiveloss  in P7 | classification_loss | regression_loss
    >
    > size contrastiveloss in P3 | size contrastiveloss in P4 | …… |size contrastiveloss  in P7 | classification_loss | regression_loss

  - 这一部分的实现还是比较简单的。

### 第四部分：优化

> 事实上，完全没有想到这一部分被覆盖了，导致这一部分开始的内容都没有了，不过没有关系，可以慢慢恢复，大致按照我的思路进行恢复吧，能恢复多少事多少。

因为显卡拿不到我现在只能够尝试对于对比学习的一些内容进行优化了，优化的可以从两个方面进行的。

#### 第一个方面：采用cuda运算

- 可能是因为没有采用cuda运算所以非常的慢，大概在7~8秒，如果按照(118287/2)*8/3600/24 = 5.47625（天）的话，这代价显然是不能够承受的，而且这单独只是对比学习的loss的计算就花费这么多少时间了。（而且测试用的一个batch的图像还是GT非常少的）

- 于是采用cuda来进行运算，然后发现时间在8秒以上。

  - > Time is  8.58696460723877

- 单独采用一个运算可能不太精确，因为cuda的加载可能比较费时，将结果运行100次之后取平均查看在cuda上的结果：

  - > Time is  0.5151817631721497

- 然后是接下来对于不在cuda上的运行结果，运行100次，时间有点久，平均下来结果：

  - > Time is  7.016696767807007

- 在cuda的运算似乎可以接受的，但是一计算之后会发现(118287/2)*0.5/3600 = 8.214375(小时)

  - 也就是说运算一个epoch光是算对比学习的loss就需要花费八个小时，好像又不太能接受了。

- 显示情况花费的时间只多不少，所以我们还需要优化。

#### 第二个方面：向量化运算

- 其实向量化运算在一开始写代码的时候就应该想到的，但是向量化运算在这里有一个很大的问题就是，因为我们计算的是不同的GT这意味着输入进来的图像矩阵是不一样规格的，也就是说不能够采取向量化运算了。

- 有一个想法，对于不够大的GT进行像素的补充（补充0不会影响结果,如下，第三个loss没有变化），然后整一个进行向量化运算的。

  - > a = torch.Size([4, 3, 928, 640])
    > Emb  torch.Size([2, 3, 134, 221]) torch.Size([2, 3, 134, 221])
    > tensor(1.3351)
    > b = Emb  torch.Size([2, 3, 569, 609]) torch.Size([2, 3, 569, 609])
    > tensor(1.1856)
    > a = Emb  torch.Size([2, 3, 716, 656]) torch.Size([2, 3, 716, 656])
    > tensor(1.3351)

- 存在另一个问题：那就是在padding每一个GT的时候其实也进行了for的循环，也就说效率也会降下来的，就是不知道会降多少了，这个得具体实验之后才之后，所以接下来就是具体的实现了。

具体实现如下：

- 第一步，找出GT里面最大的保存起来。
- 第二步，将所有的GT补0到最大的那个框。

第一步实现的代码是把图像补全到指定size 的函数，实现如下：

```python
def to_padding(a,b,old_size,new_size):
    dx , dy = new_size[0] - old_size[0],new_size[1] - old_size[1]
    l , u = dx // 2, dy // 2
    r , d = dx - l, dy - u
    pad = torch.nn.ZeroPad2d(padding=(u, d, l, r))
    return pad(a),pad(b)
```

a、b是两张用来互相对比的图片，然后old_size是原本的大小，new_size是新的大小。

接下来实现的是，循环所有的框，然后拼接起来成为一个（GT num, max width ， max height）的两个tensor

然后接下里的实现是，输入两个图像，然后把所有的GT拼接起来。

```python
def to_padding(a,b,annotation):
    annotation = annotation.long()
    size = (annotation[:,:,2] - annotation[:,:,0]) * (annotation[:,:,3] - annotation[:,:,1])
    size = torch.flatten(size)
    size_max, size_argmax = torch.max(size,dim = 0)
    index = size_argmax
    x = index // annotation.shape[0]
    y = index % annotation.shape[0]
    new_size = (annotation[x,y,2] - annotation[x,y,0],annotation[x,y,3] - annotation[x,y,1])
    img_i , img_j = np.zeros((0, a.shape[1],new_size[0],new_size[1])),np.zeros((0, a.shape[1],new_size[0],new_size[1]))
    for i in range(a.shape[0]):
        for j in annotation[i,:,:]:
            box = j
            x0,y0,x1,y1 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            embi, embj = to_padding_one(a[i,:,x0:x1,y0:y1],b[i,:,x0:x1,y0:y1],a[i,:,x0:x1,y0:y1].shape[1:],new_size)
            img_i = np.append(img_i,embi.unsqueeze(0),axis = 0) 
            img_j = np.append(img_j,embj.unsqueeze(0),axis = 0) 
            return img_i,img_j
```

经过测试，暂时没有问题的了，需要修改的一个地方是，不能够取面积最大的那个框来确定，而是应该改变取法：

- 变为width取width中最大的，height取height中最大的。

代码修改如下：

```python
def to_padding(a,b,annotation):
    annotation = annotation.long()
    width  = torch.flatten(annotation[:,:,2] - annotation[:,:,0])
    height = torch.flatten(annotation[:,:,3] - annotation[:,:,1])
    width  = torch.max(width,dim = 0)[0]
    height = torch.max(height,dim = 0)[0]

    new_size = (width,height)
    img_i , img_j = np.zeros((0, a.shape[1],new_size[0],new_size[1])),np.zeros((0, a.shape[1],new_size[0],new_size[1]))
    for i in range(a.shape[0]):
        for j in annotation[i,:,:]:
            box = j
            x0,y0,x1,y1 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            embi, embj = to_padding_one(a[i,:,x0:x1,y0:y1],b[i,:,x0:x1,y0:y1],a[i,:,x0:x1,y0:y1].shape[1:],new_size)
            img_i = np.append(img_i,embi.unsqueeze(0),axis = 0) 
            img_j = np.append(img_j,embj.unsqueeze(0),axis = 0) 
    return torch.from_numpy(img_i),torch.from_numpy(img_j)
```

然后这一部分完成了，接下来就是把其写入到retinanet/losses.py中，有一个需要注意的问题就是，一开始在ContrastiveLoss预设的batch_size必须要进行修改，变成不需要预设的输入了，需要在计算的时候动态决定了，因为每一个batch的GT数量不一定一致,但是因为其他地方也用到了，所以还是不删除这一个参数。

同时因为插入之后和原本的运行结构不一致了，需要修改forward函数的内容，

```python
class ContrastiveLoss(nn.Module):
    # def __init__(self,batch_size,n_views,f,temperature = 0.5):
    def __init__(self,batch_size,n_views,temperature = 0.5):
        super().__init__()
        self.batch_size = batch_size
        # self.f = f
        
        self.n_view = n_views
        self.register_buffer("temperature", torch.tensor(temperature))
        
    def calc_GT(self,emb_i,emb_j):
        # print("Emb ",emb_i.shape,emb_j.shape)
        batch_size = emb_i.shape[0]
        self.negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        # self.negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().cuda()
        if(emb_i.shape[1] == 0 or emb_i.shape[2] == 0):
            return torch.zeros(0.0)
        # z_i = F.normalize(emb_i, dim = 1).cuda()
        # z_j = F.normalize(emb_j, dim = 1).cuda()
        z_i = F.normalize(emb_i, dim = 1)
        z_j = F.normalize(emb_j, dim = 1)
        
        representations = torch.cat([z_i,z_j], dim = 0)
        representations = representations.view(representations.shape[0],-1)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        # print("similarity_matrix",similarity_matrix)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        # print("positives",positives)
        
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # print("loss_partial",loss_partial)
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
    def to_padding_one(a,b,old_size,new_size):
        # print(a.shape,b.shape,old_size,new_size)
        dx , dy = new_size[0] - old_size[0],new_size[1] - old_size[1]
        l , u = dx // 2, dy // 2
        r , d = dx - l, dy - u
        pad = torch.nn.ZeroPad2d(padding=(u, d, l, r))
        return pad(a),pad(b)
    def to_padding(a,b,annotation):
        annotation = annotation.long()
        width  = torch.flatten(annotation[:,:,2] - annotation[:,:,0])
        height = torch.flatten(annotation[:,:,3] - annotation[:,:,1])
        width  = torch.max(width,dim = 0)[0]
        height = torch.max(height,dim = 0)[0]

        new_size = (width,height)
        img_i , img_j = np.zeros((0, a.shape[1],new_size[0],new_size[1])),np.zeros((0, a.shape[1],new_size[0],new_size[1]))
        for i in range(a.shape[0]):
            for j in annotation[i,:,:]:
                box = j
                x0,y0,x1,y1 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                embi, embj = self.to_padding_one(a[i,:,x0:x1,y0:y1],b[i,:,x0:x1,y0:y1],a[i,:,x0:x1,y0:y1].shape[1:],new_size)
                img_i = np.append(img_i,embi.unsqueeze(0),axis = 0) 
                img_j = np.append(img_j,embj.unsqueeze(0),axis = 0) 
        return torch.from_numpy(img_i),torch.from_numpy(img_j)
    def calc_one(self,emb_i,emb_j,annotation):
        # loss = torch.tensor(0.0).cuda()
        loss = torch.tensor(0.0)
        for batch in range(self.batch_size):
            for i in annotation[batch,:,:]:
                loss += self.calc_GT(emb_i[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])],emb_j[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])])
            # print("loss0",loss)
            loss /= annotation.shape[1]
        # print("loss1",loss)
        return loss/self.batch_size
    def forward(self,features,annotations):
        new_index = torch.arange(0,self.batch_size) * self.n_view
        # emb size : [(batch size , channels , width ,height ) , …… , (batch size , channels , width ,height )]
        emb_i = [feature[new_index] for feature in features]
        emb_j = [feature[new_index+1] for feature in features]
        
        length= len(features)
        # annot size : [(batch_size, nums of GT , 5)]
        annot = [annotation[new_index] for annotation in annotations]
        contrastiveloss = torch.tensor(0.0)
        for i in range(length):
            # contrastiveloss += self.calc_one(emb_i[i].cuda(),emb_j[i].cuda(),annot[i])
            # contrastiveloss += self.calc_one(emb_i[i],emb_j[i],annot[i])
            img_i,img_j = self.to_padding(emb_i[i],emb_j[i],annot[i])
            loss += self.calc_GT(img_i,img_j)
            print("contrastiveloss",contrastiveloss)
        return contrastiveloss / length
```

完了，运行到一半已经觉得速度非常慢了，输出的结果，如下：

> Num training images: 118287
> contrastiveloss tensor(2.4368)
> contrastiveloss tensor(4.8736)
> contrastiveloss tensor(7.3105)
> contrastiveloss tensor(9.7473)
> contrastiveloss tensor(12.1841)
> Time is  30.39502787590027
> tensor(2.4368)

下面是没有使用这个方法的输出和速度：

> Num training images: 118287
> contrastiveloss tensor(0.7249)
> contrastiveloss tensor(1.4497)
> contrastiveloss tensor(2.1746)
> contrastiveloss tensor(2.8994)
> contrastiveloss tensor(3.6243)
> Time is  7.372162342071533
> tensor(0.7249)

改进方案也只是我想当然了？等一下再看一下是和cuda有没有关系，但是现在存在的一个问题就是loss不一致的问题，等等再想这个问题吧，先对比一下使用cuda和不使用cuda的速度。

不使用这个方法的时候,在运行了100次之后的结果如下：

> Time is  0.5721215438842774
> tensor(0.7249)

哦吼，循环处理GT的部分还不能够放上cuda处理，估计莫得了

> 没有结果，直接爆显存了
>
> RuntimeError: CUDA out of memory. Tried to allocate 3.08 GiB (GPU 0; 10.76 GiB total capacity; 590.49 MiB already allocated; 2.17 GiB free; 600.00 MiB reserved in total by PyTorch)

看来是没有办法向量化的了，那么loss出现的问题也就不解决了。

**这一部分是优化失败了只能希望在cuda上的计算速度足够给力了。**

### 第五部分：完成

如下是完成后的retinanet/losses.py(PS:一不小心删掉了文件功能)

```python

class ContrastiveLoss(nn.Module):
    # def __init__(self,batch_size,n_views,f,temperature = 0.5):
    def __init__(self,batch_size,n_views,temperature = 0.5):
        super().__init__()
        self.batch_size = batch_size
        # self.f = f
        
        self.n_view = n_views
        self.register_buffer("temperature", torch.tensor(temperature))
        
    def calc_GT(self,emb_i,emb_j):
        batch_size = emb_i.shape[0]
        if torch.cuda.is_available():
            self.negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().cuda()
        else:
            self.negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().cuda()
        if(emb_i.shape[1] == 0 or emb_i.shape[2] == 0):
            return torch.zeros(0.0)
        z_i = F.normalize(emb_i, dim = 1)
        z_j = F.normalize(emb_j, dim = 1)
        
        representations = torch.cat([z_i,z_j], dim = 0)
        representations = representations.view(representations.shape[0],-1)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
    def calc_one(self,emb_i,emb_j,annotation):
        if torch.cuda.is_available():
            loss = torch.tensor(0.0).cuda()
        else:
            loss = torch.tensor(0.0)
        for batch in range(self.batch_size):
            for i in annotation[batch,:,:]:
                loss += self.calc_GT(emb_i[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])],emb_j[:,:,int(i[0]):int(i[2]),int(i[1]):int(i[3])])
            loss /= annotation.shape[1]
            
        return loss/self.batch_size
    def to_padding_one(self,a,b,old_size,new_size):
        dx , dy = new_size[0] - old_size[0],new_size[1] - old_size[1]
        l , u = dx // 2, dy // 2
        r , d = dx - l, dy - u
        pad = torch.nn.ZeroPad2d(padding=(u, d, l, r))
        return pad(a),pad(b)
    def to_padding(self,a,b,annotation):
        annotation = annotation.long()
        width  = torch.flatten(annotation[:,:,2] - annotation[:,:,0])
        height = torch.flatten(annotation[:,:,3] - annotation[:,:,1])
        width  = torch.max(width,dim = 0)[0]
        height = torch.max(height,dim = 0)[0]

        new_size = (width,height)
        img_i , img_j = np.zeros((0, a.shape[1],new_size[0],new_size[1])),np.zeros((0, a.shape[1],new_size[0],new_size[1]))
        for i in range(a.shape[0]):
            for j in annotation[i,:,:]:
                box = j
                x0,y0,x1,y1 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                embi, embj = self.to_padding_one(a[i,:,x0:x1,y0:y1],b[i,:,x0:x1,y0:y1],a[i,:,x0:x1,y0:y1].shape[1:],new_size)
                img_i = np.append(img_i,embi.unsqueeze(0),axis = 0) 
                img_j = np.append(img_j,embj.unsqueeze(0),axis = 0) 
        return torch.from_numpy(img_i),torch.from_numpy(img_j)
    def forward(self,features,annotations):
        new_index = torch.arange(0,self.batch_size) * self.n_view
        emb_i = [feature[new_index] for feature in features]
        emb_j = [feature[new_index+1] for feature in features]
        
        length= len(features)
        annot = [annotation[new_index] for annotation in annotations]
        contrastiveloss = torch.tensor(0.0)
        for i in range(length):
            if torch.cuda.is_available():
                contrastiveloss += self.calc_one(emb_i[i].cuda(),emb_j[i].cuda(),annot[i])
            else:
                contrastiveloss += self.calc_one(emb_i[i],emb_j[i],annot[i])
        return contrastiveloss / length
```



未来可以做的是将每一个类别的GT都单独存起来，然后训练对比的时候，就从同类别的GT中随机抓一个拉伸之后填进对应的GT之中。

