import sys
import os
import torch
import yaml
sys.path.append('../../../../')
from models.Weight_ECFRN import Weight_ECFRN
from models.Linear_ECFRN import Linear_ECFRN
from models.Linear_ECFRN import Linear_ECFRN
from trainers.backbones import ASCO


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'CUB_fewshot_cropped/test_pre')
model_path_1 = './model_ResNet-12.pth'
model_path_2 = './model_2_ResNet-12.pth'
model_path_ASCO = './ASCO.pth'

gpu = 0
torch.cuda.set_device(gpu)

model_1 = BiFRN(resnet=True)
model_1.cuda()
model_1.load_state_dict(torch.load(model_path_1,map_location=util.get_device_map(gpu)),strict=True)
model_1.eval()

model_2 = Linear_BiFRN(resnet=True)
model_2.cuda()
model_2.load_state_dict(torch.load(model_path_2,map_location=util.get_device_map(gpu)),strict=True)
model_2.eval()

model_ASCO = ASCO
model_ASCO.cuda()
model_ASCO.load_state_dict(torch.load(model_path_ASCO,map_location=util.get_device_map(gpu)),strict=True)
model_ASCO.eval()

with torch.no_grad():
    way = 5
    for shot in [1,5]:
        phi = ASCO.predict_phi(way, shot)
        mean,interval = model_test(data_path=test_path,
                                model1=model1,
                                model2=model2,
                                phi = phi,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.2f±%.2f'%(way,shot,mean,interval))
