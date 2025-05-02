import torch
import torch.nn as nn
import utils
import torchvision
import numpy as np
import cv2

import os
from torchvision.transforms.functional import crop

def data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).cpu()
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).cpu()
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # print('-imagenet_mean-',imagenet_mean.shape)
    X = X - imagenet_mean
    X = X / imagenet_std
    return X


def inverse_data_transform(X):
    imagenet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).cpu()
    imagenet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).cpu()
    # print('-imagenet_mean-',imagenet_mean.shape)
    imagenet_mean = imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    imagenet_std = imagenet_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return torch.clip((X * imagenet_std + imagenet_mean), 0, 1)
# def data_transform(X):
#     return X


# def inverse_data_transform(X):
#     return torch.clip(X, 0, 1)

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        self.input_res = self.config.data.image_size
        self.stride = self.config.data.stride

        # self.scene_name = config.data.data_dir.split("/")[-3]
        # print(config.data.data_dir)
        # print("-scene_name-",self.scene_name)

        if os.path.isfile(args.resume):
            print("param loading in the ddm file")
            # self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()

        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='', r=None, sid = None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            # count = 132
            for i, (x, y) in enumerate(val_loader):
                # if "pavilion__Exp1_1000" in y[0]:
                
                print('-restore-',x.shape,y)

                print(self.args.image_folder, self.config.data.dataset, validation)
                print(i, x.shape, y)

                datasetname =  y[0].split('__')[0]
                fid = y[0].split('__')[1]
                frame = y[0].split('__')[2]
                print(datasetname, fid, frame)
                x_cond = x[:,:3,:,:].cpu()
                x_gt = x[:,3:,:,:].cpu()
                xs1 = x[:,3:6,:,:].clone()
                xs2 = x[:,6:9,:,:].clone()
                xs3 = x[:,9:12,:,:].clone()

                print('-x_cond-',x_cond.shape,'-x_gt-',x_gt.shape)
                print('-xs1-',xs1.shape,'-xs2-',xs2.shape,'-xs3-',xs3.shape)


                #256x256
                input_res = self.input_res
                stride = self.stride
                


                utils.logging.save_image(x_cond, os.path.join('results', validation, datasetname, 'input', fid+"_"+frame+".png"))
                # utils.logging.save_image(x_gt, os.path.join('results', validation, datasetname, 'gt', fid+"_"+frame+".png"))
                utils.logging.save_image(xs1, os.path.join('results', validation, datasetname, 'gts1', fid+"_"+frame+".png"))
                utils.logging.save_image(xs2, os.path.join('results', validation, datasetname, 'gts2', fid+"_"+frame+".png"))
                utils.logging.save_image(xs3, os.path.join('results', validation, datasetname, 'gts3', fid+"_"+frame+".png"))



                h_list = [i for i in range(0, x_gt.shape[2] - input_res + 1, stride)]
                w_list = [i for i in range(0, x_gt.shape[3] - input_res + 1, stride)]
                h_list = h_list + [x_gt.shape[2]-input_res]
                w_list = w_list + [x_gt.shape[3]-input_res]

                corners = [(i, j) for i in h_list for j in w_list]
                print('-corners-',len(corners))

                p_size = input_res
                x_grid_mask = torch.zeros_like(x_gt).cuda()
                
                print('-x_grid_mask-',x_grid_mask.shape)
                for (hi, wi) in corners:
                    # print('-hi, wi-',hi, wi)
                    x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
                x_grid_mask = x_grid_mask.cpu()
                # et_output = torch.zeros_like(x_cond).cuda()
                et_output = torch.zeros_like(x_gt).cpu()
                print('-et_output -',x_grid_mask.shape,et_output.shape)
                
                B,C,H,W = x_cond.shape

                manual_batching_size = 256
                # manual_batching_size = 3192
                # manual_batching_size = 2048
                x_cond = x_cond.cuda()

                torch.cuda.empty_cache()
                batch_size = 8192
                batches = [corners[i:i + batch_size] for i in range(0, len(corners), batch_size)]
                result_patches = []
                for batch in batches:
                    batch_patches = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for hi, wi in batch], dim=0).cpu()
                    result_patches.append(batch_patches)
                x_cond_patch = torch.cat(result_patches, dim=0)
                torch.cuda.empty_cache()
                # x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                # x_cond = x_cond.cpu()
                # x_cond_patch = x_cond_patch.cpu()
                print('-x_cond_patch1 -',x_cond_patch.shape)
                # outputall = torch.zeros_like(x_cond_patch.cpu())
                outputall = torch.zeros(
                    x_cond_patch.shape[0], 9, x_cond_patch.shape[2], x_cond_patch.shape[3],
                    device=x_cond_patch.device, dtype=x_cond_patch.dtype
                )
                for i in range(0, len(x_cond_patch), manual_batching_size):
                    print(i,i+manual_batching_size, 'using model')
                    tempoutput = self.diffusion.model( data_transform(x_cond_patch[i:i+manual_batching_size]).cuda().float() )

                    tempoutput = tempoutput.cpu()
                    outputall[i:i+manual_batching_size] = tempoutput
                    torch.cuda.empty_cache()
                print('-x_cond_patch -',x_cond_patch.shape,'-outputall -',outputall.shape)
                et_output = et_output.cuda()
                for ci in range(len(corners)):
                    hi, wi = corners[ci][0], corners[ci][1]
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += outputall[ci*B:(ci+1)*B].cuda()
                et_output = et_output.cpu()
        

                mean_output = torch.div(et_output, x_grid_mask)
                print('-mean_output -',mean_output.shape)

                # mean_output = inverse_data_transform(mean_output)
                oups1 = inverse_data_transform(mean_output[:,0:3,:,:].clone()).cpu()
                oups2 = inverse_data_transform(mean_output[:,3:6,:,:].clone()).cpu()
                oups3 = inverse_data_transform(mean_output[:,6:9,:,:].clone()).cpu()
                

                torch.cuda.empty_cache()
                utils.logging.save_image(oups1, os.path.join('results', validation, datasetname,  'outputs1',  fid+"_"+frame+".png"))
                utils.logging.save_image(oups2, os.path.join('results', validation, datasetname,  'outputs2',  fid+"_"+frame+".png"))
                utils.logging.save_image(oups3, os.path.join('results', validation, datasetname,  'outputs3',  fid+"_"+frame+".png"))

                x_cond = x_cond.cpu()
                # mean_output = mean_output.cpu()
                # x_gt = x_gt.cpu()
                # x_gt = x_gt.cpu()
                all_torchcat = torch.cat((x_cond,xs1,oups1,xs2,oups2,xs3,oups3),dim=-1)
                utils.logging.save_image(all_torchcat, os.path.join('results', validation, datasetname, 'inp_gt_oup',  fid+"_"+frame+".png"))

