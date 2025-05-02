import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
from PIL import Image
# Image.MAX_IMAGE_PIXELS = None
import re
import random
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time

import uuid

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0  # 仅在主进程运行


class polarization:
    def __init__(self, config):
        self.config = config

        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.inpsize = config.data.image_size
        self.load_memory = config.model.load_memory



    def get_loaders(self, parse_patches=True, validation=''):
        if is_main_process():
            print("=> dataset loading.")
        train_dataset = polarization_Dataset(dir=os.path.join(self.config.data.data_dir),
                                        config = self.config,
                                        patch_size=self.inpsize,
                                        n=self.config.training.patch_n,
                                        load_memory = self.load_memory,
                                        inpsize = self.inpsize,
                                        transforms=self.transforms,
                                        istrain = True,
                                        parse_patches=parse_patches)


        val_dataset = polarization_Dataset(dir=os.path.join(self.config.data.data_dir),
                                      config = self.config,
                                      patch_size=self.config.data.image_size,
                                      n=self.config.training.patch_n,
                                      load_memory = self.load_memory,
                                      inpsize = self.inpsize,
                                      transforms=self.transforms,
                                      istrain = False,
                                      parse_patches=parse_patches)

        if is_main_process():
            print("=> datasets finished.")

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        if self.config.model.ddp:
            train_sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else None
            val_sampler = DistributedSampler(val_dataset) if torch.distributed.is_initialized() else None

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                    shuffle=(train_sampler is None),
                                                    num_workers=self.config.data.num_workers,
                                                    pin_memory=True, sampler=train_sampler)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8,
                                                    shuffle=False, num_workers=self.config.data.num_workers,
                                                    pin_memory=True, sampler=val_sampler)
            if is_main_process():
                print("len(train_dataset)",len(train_dataset))
                print("len(val_dataset)",len(val_dataset))
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                    shuffle=True, num_workers=self.config.data.num_workers,
                                                    pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1 ,
                                                    shuffle=False, num_workers=self.config.data.num_workers,
                                                    pin_memory=True)

        return train_loader, val_loader


class polarization_Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, config, patch_size,n, load_memory, inpsize,  transforms, istrain=True, clip_length=3, parse_patches=True):
        super().__init__()
        if is_main_process():
            print('-dir-', dir)
            print('-self.load_memory-', load_memory)
        self.config = config
        self.istrain = istrain
        self.clip_length = clip_length
        self.dir = dir
        self.load_memory = load_memory
        input_names = []
        s1_names = []
        s2_names = []
        s3_names = []


        traindir = os.path.join(self.dir)
        # listinpdir = sorted(os.listdir(traindir))
        # for i in range(len(listinpdir)):
        if istrain:
            for idx in range(1000):
            # for idx in range(60):
                stridx = "%04d"%idx
                scene_id = os.path.join(traindir,stridx)
                input_names.append(os.path.join(scene_id,"s0.png"))
                s1_names.append(os.path.join(scene_id,"s1.png"))
                s2_names.append(os.path.join(scene_id,"s2.png"))
                s3_names.append(os.path.join(scene_id,"s3.png"))
                
        else:
            for idx in range(1000,1200):
            # for idx in range(1000,1050):
                stridx = "%04d"%idx
                scene_id = os.path.join(traindir,stridx)
                input_names.append(os.path.join(scene_id,"s0.png"))
                s1_names.append(os.path.join(scene_id,"s1.png"))         
                s2_names.append(os.path.join(scene_id,"s2.png"))        
                s3_names.append(os.path.join(scene_id,"s3.png"))        

        if is_main_process():
            print('train ori len(input_names),len(s1_names) = ', len(input_names), len(s1_names), len(s2_names), len(s3_names))
            print(input_names[0], s1_names[0], s2_names[0], s3_names[0])
            print(input_names[-1], s1_names[-1], s2_names[-1], s3_names[-1])

        self.input_names = input_names
        self.s1_names = s1_names
        self.s2_names = s2_names
        self.s3_names = s3_names

        if self.load_memory:
            self.input_matrix = []
            self.s1_matrix = []
            self.s2_matrix = []
            self.s3_matrix = []
            for i in range(len(self.input_names)):
                if is_main_process():
                    if (i+1)%100 ==0:
                        print("loading ", i)

                wd_new, ht_new = 512, 512
                input_img = PIL.Image.open(self.input_names[i]).convert("RGB")
                input_img = input_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
                s1_img = PIL.Image.open(self.s1_names[i]).convert("RGB")
                s1_img = s1_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
                s2_img = PIL.Image.open(self.s2_names[i]).convert("RGB")
                s2_img = s2_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
                s3_img = PIL.Image.open(self.s3_names[i]).convert("RGB")
                s3_img = s3_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
                self.input_matrix.append(input_img)
                self.s1_matrix.append(s1_img)
                self.s2_matrix.append(s2_img)
                self.s3_matrix.append(s3_img)


        self.patch_size = patch_size
        self.inpsize = inpsize
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        # print('-self.patch_size self.n -',self.patch_size, self.n)


    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0], [0], h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        if self.config.model.ddp:
            rank = dist.get_rank()
            unique_part = uuid.uuid4().int & (1<<16)-1  
            seed = 42 + rank + int(time.time() * 1000) + unique_part
            random.seed(seed)
            index = random.randint(0, len(self.input_names)-1)

        input_name = self.input_names[index]
        s1_names = self.s1_names[index]
        s2_names = self.s2_names[index]
        s3_names = self.s3_names[index]
        datasetname = re.split('/', input_name)[-4]
        img_vid = re.split('/', input_name)[-2]
        img_id = re.split('/', input_name)[-1][:-4]
        img_id = datasetname + '__' + img_vid + '__' + img_id
        if  self.load_memory:
            input_img = self.input_matrix[index]
            s1_img = self.s1_matrix[index]
            s2_img = self.s2_matrix[index]
            s3_img = self.s3_matrix[index]
        else:
            input_img = PIL.Image.open(input_name).convert("RGB")
            s1_img = PIL.Image.open(s1_names).convert("RGB")
            s2_img = PIL.Image.open(s2_names).convert("RGB")
            s3_img = PIL.Image.open(s3_names).convert("RGB")

            wd_new, ht_new = 512, 512
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
            s1_img = s1_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
            s2_img = s2_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
            s3_img = s3_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)


        if self.parse_patches:


            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            s1_img = self.n_random_crops(s1_img, i, j, h, w)
            s2_img = self.n_random_crops(s2_img, i, j, h, w)
            s3_img = self.n_random_crops(s3_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), 
                                  self.transforms(s1_img[i]),
                                  self.transforms(s2_img[i]),
                                  self.transforms(s3_img[i])
                                  ], dim=0)
                       for i in range(self.n)]
            # return torch.stack(outputs, dim=0), img_id
            outputs = torch.stack(outputs, dim=0)
            # print('outputs',outputs.shape)


            return outputs, img_id


        else:
            wd_new, ht_new = input_img.size
            # wd_new = 512
            # ht_new = 512
            wd_new = 256
            ht_new = 256
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
            s1_img = s1_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
            s2_img = s2_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)
            s3_img = s3_img.resize((wd_new, ht_new), PIL.Image.BILINEAR)

            return torch.cat([self.transforms(input_img), 
                              self.transforms(s1_img),
                              self.transforms(s2_img),
                              self.transforms(s3_img)
                              ], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
