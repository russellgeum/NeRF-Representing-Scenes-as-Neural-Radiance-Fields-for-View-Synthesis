import os
import sys
import glob
from tqdm import tqdm
from tqdm import trange
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_loader import *
from models import *
from parser import *



DEBUG = False
np.random.seed(0)



class Setting(object):
    def __init__(self, args, device):
        self.args   = args
        self.device = device

        ## Create dataset
        self.setting = {}
        self.i_batch, self.data_dict = self.load_dataset()

        ## Create nerf model
        parameters = []
        self.neural_rendering = NeuralRendering(self.args, self.device)
        parameters += list(self.neural_rendering.model.parameters())
        parameters += list(self.neural_rendering.model_fine.parameters())

        ## Create optimizer
        self.optim = torch.optim.Adam(params = parameters, lr = self.args.lrate, betas = (0.9, 0.999))

        self.setting["i_batch"] = self.i_batch
        self.setting["data"]    = self.data_dict
        self.setting["model"]   = self.neural_rendering
        self.setting["optim"]   = self.optim


        self.kwargs_train = {
            'N_samples' :     self.args.N_samples,
            'N_importance' :  self.args.N_importance,
            'perturb' :       self.args.perturb,
            'white_bkgd' :    self.args.white_bkgd,
            'use_viewdirs' :  self.args.use_viewdirs,
            'raw_noise_std' : self.args.raw_noise_std}

        if self.args.dataset_type != 'llff' or self.args.no_ndc: ## NDC only good for LLFF-style forward facing data
            print('Not ndc!')
            self.kwargs_train['ndc']     = False
            self.kwargs_train['lindisp'] = self.args.lindisp

        self.kwargs_test                  = {k : self.kwargs_train[k] for k in self.kwargs_train}
        self.kwargs_test['perturb']       = False
        self.kwargs_test['raw_noise_std'] = 0.

        self.kwargs_train.update({'near' : self.data_dict["near"], 'far' : self.data_dict["far"]})
        self.kwargs_test.update({'near' : self.data_dict["near"], 'far' : self.data_dict["far"]})

        self.setting["kwargs_train"] = self.kwargs_train
        self.setting["kwargs_test"]  = self.kwargs_test


    def load_dataset(self):
        if self.args.dataset_type == 'llff':
            i_batch   = 0
            data_dict = LLFFDataset(self.args, self.device)
        elif self.args.dataset_type == 'blender':
            i_batch   = None
            data_dict = BlenderDataset(self.args, self.device)
        else:
            print('Unknown dataset type', self.args.dataset_type, 'exiting')
            return
        return i_batch, data_dict


    @staticmethod
    def rendering_test(render, render_poses, hwf, K, chunk, kwargs, savedir = None, render_factor = 0):
        H, W, focal = hwf

        if render_factor!=0: # Render downsampled for speed
            H     = H // render_factor
            W     = W // render_factor
            focal = focal / render_factor

        rgbs  = []
        disps = []
        for i, c2w in enumerate(tqdm(render_poses)):
            rgb, disp, _, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **kwargs)
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())
            if i==0:
                print(rgb.shape, disp.shape)

            if savedir is not None:
                rgb8     = image2uint(rgbs[-1])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

        rgbs  = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)
        return rgbs, disps