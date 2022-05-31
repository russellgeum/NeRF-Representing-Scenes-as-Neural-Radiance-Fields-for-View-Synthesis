import os
import sys
import time
import json
import random

from tqdm import tqdm
from tqdm import trange
import imageio
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_loader import *
from parser import *
from models import *
from tools import *




class Trainer():
    def __init__(self):
        parser      = config_parser()
        self.args   = parser.parse_args()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(">>> learning_rate   ", self.args.lrate)
        print(">>> learning_decay  ", self.args.lrate_decay)
        print(">>> Dataset: {} -> use_batching: {} ".format(self.args.dataset_type, self.args.use_batching))

        print(">>> N_rand          ", self.args.N_rand)
        print(">>> N_samples       ", self.args.N_samples)
        print(">>> N_importance    ", self.args.N_importance)
        print(">>> multires        ", self.args.multires)
        print(">>> multires_views  ", self.args.multires_views)
        print(">>> {}, {}".format(self.args.chunk, self.args.netchunk))
        
        print(">>> perturb         ", self.args.perturb)
        print(">>> noise_std       ", self.args.raw_noise_std)
        print(">>> factor          ", self.args.factor)
        print(">>> llffhold        ", self.args.llffhold)
        print(">>> lindisp         ", self.args.lindisp)
        print(">>> spherify        ", self.args.spherify)
        print(">>> white_bkgd      ", self.args.white_bkgd)

        print(">>> precrop_iters   ", self.args.precrop_iters)
        print(">>> precrop_frac    ", self.args.precrop_frac)
        print(">>> half_resolution ", self.args.half_res)

        ## Create log dir and copy the config file
        self.basedir      = self.args.basedir
        self.expname      = self.args.expname
        self.N_rand       = self.args.N_rand
        self.use_batching = self.args.use_batching ## no_batching F -> use_batching T && no_batching T -> use_batching F

        os.makedirs(os.path.join(self.basedir, self.expname), exist_ok=True)
        with open(os.path.join(self.basedir, self.expname, 'self.args.txt'), 'w') as file:
            for arg in sorted(vars(self.args)):
                attr = getattr(self.args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self.args.config is not None:
            with open(os.path.join(self.basedir, self.expname, 'config.txt'), 'w') as file:
                file.write(open(self.args.config, 'r').read())

        self.start   = 0
        self.model_config = Setting(self.args, self.device)
        
        self.i_batch = self.model_config.setting["i_batch"]
        self.kwargs_train = self.model_config.setting["kwargs_train"]
        self.kwargs_test  = self.model_config.setting["kwargs_test"]
        self.H   = self.model_config.setting["data"]["H"]
        self.W   = self.model_config.setting["data"]["W"]
        self.K   = self.model_config.setting["data"]["K"]
        self.hwf = self.model_config.setting["data"]["hwf"]


        # ## Short circuit if only rendering out from trained model
        # if args.render_only:
        #     print('RENDER ONLY')
        #     with torch.no_grad():
        #         if args.render_test:
        #             images = images[self.model_config.setting["data"]["i_test"]]
        #         else:
        #             images = None

        #         testsavedir = os.path.join(self.basedir, self.expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        #         os.makedirs(testsavedir, exist_ok=True)
        #         print('test poses shape', self.model_config.setting["data"]["render_poses"].shape)

        #         rgbs, _ = rendering_test(
        #             self.model_config.setting["data"]["render_poses"], self.model_config.setting["data"]["hwf"], self.model_config.setting["data"]["K"], 
        #             args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        #         print('Done rendering', testsavedir)
        #         imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
        #         return

    
    def iteration(self):
        self.global_step = self.start
        start            = self.start + 1
        N_iters          = 200000 + 1
        for batch_idx in trange(start, N_iters):
            if self.use_batching:
                batch        = self.model_config.setting["data"]["rays_rgb"][self.i_batch: self.i_batch + self.N_rand] # [B, 2+1, 3*?]
                batch        = torch.transpose(batch, 0, 1)
                batch_rays   = batch[:2]
                target_image = batch[2]

                self.i_batch += self.N_rand
                if self.i_batch >= self.model_config.setting["data"]["rays_rgb"].shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx              = torch.randperm(self.model_config.setting["data"]["rays_rgb"].shape[0])
                    self.model_config.setting["data"]["rays_rgb"] = self.model_config.setting["data"]["rays_rgb"][rand_idx]
                    self.i_batch = 0

            else:
                random_idx = np.random.choice(self.model_config.setting["data"]["i_train"])
                target     = self.model_config.setting["data"]["images"][random_idx]
                target     = torch.Tensor(target).to(self.device)
                pose       = self.model_config.setting["data"]["poses"][random_idx, :3,:4]

                if self.N_rand is not None: ## (H, W, 3), (H, W, 3)
                    rays_o, rays_d = compute_rays(self.H, self.W, self.K, torch.Tensor(pose), "torch")

                    if batch_idx < self.args.precrop_iters:
                        dH     = int(self.H//2 * self.args.precrop_frac)
                        dW     = int(self.W//2 * self.args.precrop_frac)
                        grid_H = torch.linspace(self.H//2 - dH, self.H//2 + dH - 1, 2*dH)
                        grid_W = torch.linspace(self.W//2 - dW, self.W//2 + dW - 1, 2*dW)
                        coords = torch.stack(torch.meshgrid(grid_H, grid_W), -1)
                        if batch_idx == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.args.precrop_iters}")                
                    else: ## (H, W, 2)
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, self.H-1, self.H), torch.linspace(0, self.W-1, self.W)), -1)

                    coords        = torch.reshape(coords, [-1,2])  # (H * W, 2)
                    select_inds   = np.random.choice(coords.shape[0], size=[self.N_rand], replace = False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)

                    rays_o        = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d        = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays    = torch.stack([rays_o, rays_d], 0)
                    target_image  = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            rgb, _, _, extras = self.model_config.setting["model"](
                self.H, self.W, self.K, chunk = self.args.chunk, rays = batch_rays, retraw = True, **self.kwargs_train)

            loss, psnr = self.parameter_update(rgb, extras, target_image)


            # Rest is logging
            if batch_idx % self.args.i_weights==0:
                save_path = os.path.join(self.basedir, self.expname, '{:06d}.tar'.format(batch_idx))
                torch.save({
                    'global_step': self.global_step,
                    'model_state_dict': self.model_config.setting["model"].model.state_dict(),
                    'model_fine_state_dict': self.model_config.setting["model"].model_fine.state_dict(),
                    'optimizer_state_dict': self.model_config.setting["optim"].state_dict(),}, save_path)
                print('Saved checkpoints at', save_path)


            if batch_idx % self.args.i_video==0 and batch_idx > 0:
                with torch.no_grad():
                    rgbs, disps = self.model_config.rendering_test(
                        self.model_config.setting["model"], self.model_config.setting["data"]["render_poses"], self.hwf, self.K, self.args.chunk, self.kwargs_test)

                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(self.basedir, self.expname, '{}_spiral_{:06d}_'.format(self.expname, batch_idx))
                imageio.mimwrite(moviebase + 'rgb.mp4', image2uint(rgbs), fps = 30, quality = 8)
                imageio.mimwrite(moviebase + 'disp.mp4', image2uint(disps / np.max(disps)), fps = 30, quality = 8)

                # if args.use_viewdirs:
                #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                #     with torch.no_grad():
                #         rgbs_still, _ = rendering_test(render_poses, hwf, args.chunk, render_kwargs_test)
                #     render_kwargs_test['c2w_staticcam'] = None
                #     imageio.mimwrite(moviebase + 'rgb_still.mp4', image2uint(rgbs_still), fps=30, quality=8)

            if batch_idx % self.args.i_testset==0 and batch_idx > 0:
                testsavedir = os.path.join(self.basedir, self.expname, 'testset_{:06d}'.format(batch_idx))
                os.makedirs(testsavedir, exist_ok=True)
                with torch.no_grad():
                    test_poses = torch.Tensor(self.model_config.setting["data"]["poses"][self.model_config.setting["data"]["i_test"]])
                    test_poses = test_poses.to(self.device)
                    self.model_config.rendering_test(
                        self.model_config.setting["model"], test_poses, self.hwf, self.K, self.args.chunk, self.kwargs_test, savedir = testsavedir)
                print('Saved test set')


            if batch_idx % self.args.i_print==0:
                tqdm.write(f">>>   [TRAIN] Iter: {batch_idx} Loss: {loss.item()}  PSNR: {psnr.item()}")
            self.global_step += 1


    def parameter_update(self, outputs, extras, target_image):
        self.model_config.setting["optim"].zero_grad()

        loss0    = MSELoss(outputs, target_image)
        loss     = loss0
        psnr     = PSNR(loss0)
        if 'rgb0' in extras:
            loss1 = MSELoss(extras['rgb0'], target_image)
            loss  = loss + loss1

        loss.backward()
        self.model_config.setting["optim"].step()

        decay_rate  = 0.1
        decay_steps = self.args.lrate_decay * 1000
        new_lrate   = self.args.lrate * (decay_rate ** (self.global_step / decay_steps))
        for param_group in self.model_config.setting["optim"].param_groups:
            param_group['lr'] = new_lrate

        return loss, psnr




if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train = Trainer()
    train.iteration()