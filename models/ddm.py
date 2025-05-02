import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import random
import torchvision.utils as tvu
import os
import torch.nn.functional as F
import matplotlib
import cv2

from models.Uformer import create_uformer_nets
from models.restormer import create_restormer_nets
from models.models_mae import mae_vit_large_patch16_dec512d8b
# from models.models_mae_adaptor import mae_vit_large_patch16_dec512d8b

from torchvision.transforms.functional import crop
import random
import torch.optim as optim
import torch.nn.functional as F




def data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(X.device)
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(X.device)
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # print('-imagenet_mean-',imagenet_mean.shape,X.shape)
    X = X - imagenet_mean
    X = X / imagenet_std
    return X


def inverse_data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).to(X.device)
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).to(X.device)
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return torch.clip((X * imagenet_std + imagenet_mean), 0, 1)


class EMAHelper(object):
    # def __init__(self, mu=0.9999):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                # param_device = param.data.device
                # self.shadow[name] = param.data.clone().to(param_device)

    def update(self, module):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                # param_device = param.data.device
                # print('-param_device-',param_device)
                # self.shadow[name] = self.shadow[name].to(param_device)
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

                # self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = torch.nn.parallel.DistributedDataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def params_count(net):
    list1 = []
    for p in net.parameters():
        # print('p-',p.shape)
        list1.append(p)
    # print('len(net.parameters)',len(list1))
    n_parameters = sum(p.numel() for p in net.parameters())
    print('-----Model param: {:.5f}M'.format(n_parameters / 1e6))
    # print('-----Model memory: {:.5f}M'.format(n_parameters/1e6))
    return n_parameters






class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        isddp = config.model.ddp


        if isddp:
            # Initialize the process group
            dist.init_process_group("nccl")
            assert config.training.batch_size % dist.get_world_size() == 0
            self.rank = dist.get_rank()
            local_rank = int(os.getenv('LOCAL_RANK', self.rank % torch.cuda.device_count()))
            torch.cuda.set_device(local_rank)  # Set the correct GPU for this process
            print("-local_rank-",local_rank)
            self.device = local_rank


        # self.model = create_restormer_nets()
        # self.model_name = 'Restormer'

        # self.model = create_uformer_nets()
        # self.model.to(memory_format=torch.contiguous_format)
        # self.model_name = 'Uformer'

        self.model = mae_vit_large_patch16_dec512d8b()
        self.model_name = 'MAE'
        param_path = './mae_visualize_vit_large_ganloss.pth'
        param_path = torch.load(param_path)
        old_weight = param_path['model']['decoder_pred.weight']  # shape: [768, 512]
        old_bias = param_path['model']['decoder_pred.bias']      # shape: [768]
        print("old_weight",old_weight.shape,"old_bias",old_bias.shape )
        new_weight = old_weight.repeat(3, 1)  # shape: [2304, 512]
        new_bias = old_bias.repeat(3)         # shape: [2304]
        param_path['model']['decoder_pred.weight'] = new_weight
        param_path['model']['decoder_pred.bias'] = new_bias

        print("-model_name-",self.model_name)
        if isddp:
            if self.rank == 0:
                params_count(self.model)
        else:
            params_count(self.model)


    
        self.start_epoch, self.step = 0, 0
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)


        # self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                        betas=(0.9, 0.999), amsgrad=self.config.optim.amsgrad, eps=self.config.optim.eps)
           
        if isddp:
            self.model = DDP(self.model.to(self.device), device_ids=[self.rank], find_unused_parameters=False,gradient_as_bucket_view=True)
        else:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)




    def load_ddm_ckpt(self, load_path, ema=False):
        # checkpoint = utils.logging.load_checkpoint(load_path, None)
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        # self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])

        # print("=> loaded checkpoint '{}' )".format(load_path))

        print("=> loaded checkpoint '{}' (step {})".format(load_path, self.step))




    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        # if os.path.isfile(self.args.resume):
        #     self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):

            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                # print(i,x.shape,y)
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                # print('input',x.shape)
                self.model.train()

                xinp = x[:,0:3,:,:].clone()
                xs1 = x[:,3:6,:,:].clone()
                xs2 = x[:,6:9,:,:].clone()
                xs3 = x[:,9:12,:,:].clone()
                # print('-xinp-',xinp.shape,'-xs1-',xs1.shape,'-xs2-',xs2.shape,'-xs3-',xs3.shape)

                xinp = xinp.to(self.device)
                xs1 = xs1.to(self.device)
                xs2 = xs2.to(self.device)
                xs3 = xs3.to(self.device)

                B,C,H,W = xinp.shape

                xinp = data_transform(xinp).float().contiguous()
                xs1 = data_transform(xs1).float().contiguous()
                xs2 = data_transform(xs2).float().contiguous()
                xs3 = data_transform(xs3).float().contiguous()
                xgt = torch.cat((xs1,xs2,xs3),dim=1)



                self.model.train()
                xoup = self.model(xinp)
                # print(str(self.step), '-xinp-',xinp.shape,'-xgt-',xgt.shape,'-xoup-',xoup.shape)
  

                loss1 = (xoup - xgt)

                loss1 = torch.abs(loss1).mean(dim=-1).mean(dim=-1).mean(dim=-1).mean(dim=-1)

                loss = loss1

                if self.step % 10 == 0:
                    if self.rank == 0:
                        print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    
                    if self.rank == 0 or self.rank == 1:
                        with torch.no_grad():
                            lenvis = 5 
                            # print('-split_region-',split_region.shape)
                            vis_xinp = inverse_data_transform(xinp[:lenvis])
                            gts1 = inverse_data_transform(xgt[:lenvis,0:3,:,:].clone())
                            gts2 = inverse_data_transform(xgt[:lenvis,3:6,:,:].clone())
                            gts3 = inverse_data_transform(xgt[:lenvis,6:9,:,:].clone())

                            oups1 = inverse_data_transform(xoup[:lenvis,0:3,:,:].clone())
                            oups2 = inverse_data_transform(xoup[:lenvis,3:6,:,:].clone())
                            oups3 = inverse_data_transform(xoup[:lenvis,6:9,:,:].clone())

                            image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size)+"_"+self.model_name, str(self.step))
                            os.makedirs(image_folder, exist_ok=True)
                            for io in range(len(xinp[:lenvis])):
                                all_torchcat = torch.cat( (vis_xinp[io:io+1,:,:,:],
                                                           gts1[io:io+1,:,:,:], oups1[io:io+1,:,:,:],
                                                           gts2[io:io+1,:,:,:], oups2[io:io+1,:,:,:],
                                                           gts3[io:io+1,:,:,:], oups3[io:io+1,:,:,:]
                                                           ),dim=-1)
                                tvu.save_image(all_torchcat, os.path.join(image_folder, "train_"+"node"+str(self.rank)+"_"+f"{io}_inp_gt_oup.jpg"))
                    self.sample_validation_patches(val_loader, self.step)

                if self.step % self.config.training.snapshot_freq == 0:
                    if self.rank == 0:
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'state_dict': self.model.module.state_dict(),
                        }, filename=os.path.join('param/'+self.model_name+'/', self.config.data.dataset))
                    dist.barrier()

                self.step += 1
    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            # print('-sample_image-',x.shape, x_cond.shape)
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        # image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size)+"_"+self.model_name, str(self.step))

        with torch.no_grad():
            if self.rank == 0:
                print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            # print("x shape",x.shape)


            x_cond = x[:,0:3,:,:].clone().to(self.device)
            x_cond = data_transform(x_cond).float()

            gts1 = x[:,3:6,:,:].clone().cpu()
            gts2 = x[:,6:9,:,:].clone().cpu()
            gts3 = x[:,9:12,:,:].clone().cpu()


            xoup = self.model(x_cond)
            if self.rank == 0:
                print('-test x_cond-',x_cond.shape,'-test xoup-',xoup.shape)
            oups1 = inverse_data_transform(xoup[:,0:3,:,:].clone()).cpu()
            oups2 = inverse_data_transform(xoup[:,3:6,:,:].clone()).cpu()
            oups3 = inverse_data_transform(xoup[:,6:9,:,:].clone()).cpu()
            # print('-split_region-',split_region.shape)
            vis_xinp = inverse_data_transform(x_cond).cpu()
            if self.rank == 0:
                for io in range(len(x)):
                    all_torchcat = torch.cat( (vis_xinp[io:io+1,:,:,:],
                                                gts1[io:io+1,:,:,:], oups1[io:io+1,:,:,:],
                                                gts2[io:io+1,:,:,:], oups2[io:io+1,:,:,:],
                                                gts3[io:io+1,:,:,:], oups3[io:io+1,:,:,:]
                                                ),dim=-1)
                    tvu.save_image(all_torchcat, os.path.join(image_folder, "test_"+f"{io}_real_inp_oup.png"))
