import torch
import torch.nn as nn
from torchvision.models import resnet50

'''ResNet50'''
resnet = resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
# 2 - last two layers we tweak

'''mobilenetv3'''
from torchvision.models import mobilenet_v3_large
mobilenet = mobilenet_v3_large(pretrained=True)
mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)



