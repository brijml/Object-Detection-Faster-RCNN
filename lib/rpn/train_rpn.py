import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16

vgg16 = vgg16()
print vgg16