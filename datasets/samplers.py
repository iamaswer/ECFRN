import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

# sampler used for meta-training
class meta_batchsampler(Sampler):

    # 初始化方法，接收数据源、类别数和样本数作为参数
    def __init__(self,data_source,way,shots):

        self.way = way
        self.shots = shots

        class2id = {}  # 存储类别到样本索引的映射

        #  遍历数据源，构建类别到样本索引的映射
        #  遍历 data_source.imgs 列表中的每个元素，每个元素包含一个图片路径和一个类别标签，同时获取它们的索引值、图片路径和类别标签
        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id] = []    # 确保class_id在class2id中
            class2id[class_id].append(i)  # 将i插入到class2id中，插入的位置是class_id

        self.class2id = class2id  # 保存类别到样本索引的映射


    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        # import sys
        # print("temp_class2id", type(temp_class2id), temp_class2id.shape)
        # sys.exit()
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])  # 对每个类别的样本索引进行随机化

        # 循环生成元批次数据，直到剩余类别数小于元批次中的类别数
        while len(temp_class2id) >= self.way:

            id_list = []  # 存储元批次中的样本索引

            list_class_id = list(temp_class2id.keys())  # 获取类别列表

            # 计算每个类别的样本数，用于设置采样概率
            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])

            # 随机选择一些类别作为元批次中的类别，并根据采样概率选择
            batch_class_id = np.random.choice(list_class_id,size=self.way,replace=False,p=pcount/sum(pcount))

            # 根据每个类别的样本数和样本索引列表，生成元批次中的样本索引列表
            for shot in self.shots:
                for class_id in batch_class_id:
                    for _ in range(shot):
                        id_list.append(temp_class2id[class_id].pop())

            # 移除已经使用过的类别，确保下一个元批次中不会重复使用
            for class_id in batch_class_id:
                if len(temp_class2id[class_id])<sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list


# sampler used for meta-testing
class random_sampler(Sampler):

    def __init__(self,data_source,way,shot,query_shot=16,trial=1000):

        class2id = {}

        for i,(image_path,class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id]=[]
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.query_shot = query_shot

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot
        
        class2id = deepcopy(self.class2id)        
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []
 
            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])
                
            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot+query_shot)])

            yield id_list