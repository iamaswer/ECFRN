import os
import sys
import torch
import yaml
import numpy as np
from functools import partial
sys.path.append('../../../../')
from trainers import trainer, bifrn_train
from datasets import dataloaders
from models.backbones.ASCO import ASCO
from models.Weight_ECFRN  import Weight_ECFRN
from utils import util

with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
path = os.path.abspath(temp['data_path'])

data_path = os.path.join(path,'dogs/train')
save_path  = 'model_asco'
# Pre-trained model stub
model_path_1 = './model1_Conv-4.pth'

gpu = 0
torch.cuda.set_device(gpu)

model1 = Weight_ECFRN(resnet=False)
model1.cuda()
model1.load_state_dict(torch.load(model_path_1,map_location=util.get_device_map(gpu)),strict=True)
model1.eval()

model = ASCO(pretrained_model=model1,
             data_path=data_path,)
torch.save(model.state_dict(),save_path)
# Initialize reconstruction module

# Prepare experimental dataset with sample C and K values
C_values = np.arange(2, 5)  # Example: C = [2, 3, ..., 10]
K_values = np.arange(1, 5)  # Example: K = [1, 2, ..., 5]
model.prepare_experimental_dataset(C_values, K_values, pre=False, transform_type=0)
print(model.Dexp)
# Train the linear model
model.train_model()
torch.save(model.state_dict(),save_path)
# Predict phi for a specific combination of C and K

