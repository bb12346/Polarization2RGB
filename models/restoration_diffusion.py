import torch
import torch.nn as nn
import utils
import torchvision
import numpy as np
import cv2

import os
from torchvision.transforms.functional import crop

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def train_inference(x, x_cond, seq, diffusion, b, eta=0., corners=None, p_size=None, manual_batching=False):
    # print('-train_inference-',x.shape,x_cond.shape)
    # print("diffusion",diffusion.model)
    with torch.no_grad():


        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros_like(x).cuda()
        for ( hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
        count = 0
        # print(x_grid_mask.shape)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # print(i,j,count)
            count = count + 1

            n = x.size(0)

            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            xt = xs[-1].cuda()
            x_cond = x_cond.cuda()
            et_output = torch.zeros_like(x).cuda()
            # print("et_output",et_output.shape)
            xt_patch = torch.cat([xt[:, :, hi:hi + p_size, wi:wi + p_size] for (hi, wi) in corners], dim=0)
            x_cond_patch = torch.cat([data_transform(x_cond[:, :, hi:hi + p_size, wi:wi + p_size]) for (hi, wi) in corners], dim=0)



            manual_batching_size = 1024

            for idx in range(0, len(corners), manual_batching_size):
                idx_bs = torch.cat([x_cond_patch[idx:idx+manual_batching_size], 
                                            xt_patch[idx:idx+manual_batching_size]], dim=1)
                t_bs = t.repeat(idx_bs.size(0))
                # print('-idx_bs-',idx_bs.shape,t_bs.shape)
                outputs = diffusion.model(idx_bs, t_bs)
                # print('-idx outputs-',outputs.shape)
                for idx, (hi, wi) in enumerate(corners[idx:idx+manual_batching_size]):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]

            # print('-et outputs-',et_output.shape)
            et = torch.div(et_output, x_grid_mask)
            # print('-et -',et.shape,'-xt-',xt.shape)
            # print('-at -',at)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            # print('-x0_t -',x0_t.shape)

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            # print('-xt_next -',xt_next.shape)
            xs.append(xt_next.to('cpu'))
    # print('-output-',len(xs),len(x0_preds))
    return xs, x0_preds

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        self.input_res = self.config.data.image_size
        self.stride = self.config.data.stride
        self.step = self.config.data.step

        # self.scene_name = config.data.data_dir.split("/")[-3]
        # print(config.data.data_dir)
        # print("-scene_name-",self.scene_name)

        if os.path.isfile(args.resume):
            print("param has loaded in ddm.py")
            # self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()

        else:
            print('Pre-trained diffusion model path is missing!')


    def get_result(self, config, random_noise, x_cond, x_gt, diffusion, input_res, stride, step=25):
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().cuda()
        skip = self.config.diffusion.num_diffusion_timesteps // step
        # skip = self.config.diffusion.num_diffusion_timesteps // 8
        # skip = self.config.diffusion.num_diffusion_timesteps // 5
        # skip = self.config.diffusion.num_diffusion_timesteps // 3
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)


        h_list = [i for i in range(0, x_gt.shape[2] - input_res + 1, stride)]
        w_list = [i for i in range(0, x_gt.shape[3] - input_res + 1, stride)]
        if x_gt.shape[2] % input_res !=0:
            h_list = h_list + [x_gt.shape[2]-input_res]
        if x_gt.shape[3] % input_res !=0:
            w_list = w_list + [x_gt.shape[3]-input_res]
        corners = [(i, j) for i in h_list for j in w_list]

        print("len(corners)", len(corners))
        xs = train_inference(random_noise, x_cond, seq, diffusion, self.betas, eta=0., corners=corners, p_size=self.config.data.image_size, manual_batching=True)
        pseudogt = xs[0][-1]
        # pseudogt = xs[1][-1]
        # pseudogt = inverse_data_transform(pseudogt)

        return pseudogt


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
                step = self.step
                
                utils.logging.save_image(x_cond, os.path.join('results', validation, datasetname, 'input', fid+"_"+frame+".png"))
                # utils.logging.save_image(x_gt, os.path.join('results', validation, datasetname, 'gt', fid+"_"+frame+".png"))
                utils.logging.save_image(xs1, os.path.join('results', validation, datasetname, 'gts1', fid+"_"+frame+".png"))
                utils.logging.save_image(xs2, os.path.join('results', validation, datasetname, 'gts2', fid+"_"+frame+".png"))
                utils.logging.save_image(xs3, os.path.join('results', validation, datasetname, 'gts3', fid+"_"+frame+".png"))

                random_noise = torch.randn(x_gt.size()).cuda()
                mean_output = self.get_result(self.config, random_noise, x_cond, x_gt, self.diffusion, input_res, stride, step)
                print("mean_output",mean_output.shape)
                
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

