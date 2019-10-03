print('starting')
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import torch.nn as nn
import os
import numpy as np
import json
from PIL import Image

from functools import partial
import pickle

class myResNet(nn.Module):
    def __init__(self, orig_resnet):
        super(myResNet, self).__init__()
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool

        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.avgpool = orig_resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

print('starting')

# th architecture to use
arch = 'resnet50'
train_odgt = '/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/data/training.odgt'
val_odgt = '/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/data/validation.odgt'
data_root = '/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/data/'
train_output = '/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/feature/feat_train.npy'
val_output = '/mnt/lustre/zhaoyinan/semantic-segmentation-pytorch/feature/feat_val.npy'
# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch

model = models.__dict__[arch](num_classes=365)
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

myModel = myResNet(model)

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print('loading json')
train_list_sample = [json.loads(x.rstrip()) for x in open(train_odgt, 'r')]
val_list_sample = [json.loads(x.rstrip()) for x in open(val_odgt, 'r')]

feat_train = np.zeros([len(train_list_sample), 2048])
feat_val = np.zeros([len(val_list_sample), 2048])

print('entering loop')

for i in range(len(train_list_sample)):
    if i%100 == 0:
        print(i)
    this_sample = train_list_sample[i]
    # load the test image
    img_name = data_root + this_sample['fpath_img']
    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    feat = myModel.forward(input_img)
    feat_train[i] = feat[0].detach().numpy()

np.save(train_output, feat_train)
    
for i in range(len(val_list_sample)):
    if i%100 == 0:
        print(i)
    this_sample = val_list_sample[i]
    # load the test image
    img_name = data_root + this_sample['fpath_img']
    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    feat = myModel.forward(input_img)
    feat_val[i] = feat[0].detach().numpy()

np.save(val_output, feat_val)



