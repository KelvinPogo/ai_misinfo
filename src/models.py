import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

'''ResNet50'''
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
# 2 - last two layers we tweak

'''mobilenetv3'''
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)



