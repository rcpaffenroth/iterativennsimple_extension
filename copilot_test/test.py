# here is a nerual network with 17 layers for doing the MNIST classification problem
# the network is trained with the MNIST dataset and the trained model is saved in the file "model.h5"
# be sure to use pytorch since tensorflow sucks
# the code is written in python3
# the code is written by Yiming Liu


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import matplotlib.pyplot as plt
import math
import time
import random
import sys
import pickle
import gzip

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

from PIL import Image