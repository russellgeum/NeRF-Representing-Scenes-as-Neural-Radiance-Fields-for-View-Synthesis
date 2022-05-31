import os
import json
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from models import *




trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()


rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()


rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()



def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w



def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas  = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)


    all_images = []
    all_poses  = []
    counts     = [0]
    for s in splits:
        meta  = metas[s]
        imgs  = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs  = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_images.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    imgs    = np.concatenate(all_images, 0)
    poses   = np.concatenate(all_poses, 0)
    H, W    = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal   = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack(
        [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[ :-1]], 0)
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

        
    return imgs, poses, [H, W, focal], render_poses, i_split




def BlenderDataset(args, device):
    images, poses, hwf, render_poses, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_valid, i_test = i_split

    near = 2.
    far  = 6.
    print('NEAR FAR', near, far)
    if args.white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]


    if args.render_test:
        render_poses = np.array(poses[i_test])

    ## Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf  = [H, W, focal]
    K    = np.array([
        [focal, 0,    0.5*W],
        [0,     focal, 0.5*H],
        [0,     0,     1]])


    if args.use_batching:
        print("Prepare raybatch tensor if batching random rays")
        rays     = np.stack([compute_rays(H, W, K, p, "numpy") for p in poses[:,:3,:4]]) # [N, ro+rd, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        np.random.shuffle(rays_rgb)


    ## Move training data to GPU
    images       = torch.Tensor(images)
    poses        = torch.Tensor(poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)
    input_dict   = {
        "images":       images,
        "poses":        poses,
        "render_poses": render_poses,
        "near": near,
        "far":  far,
        "hwf":  hwf,
        "H": H,
        "W": W,
        "K": K,
        "i_train": i_train,
        "i_valid": i_valid,
        "i_test":  i_test,
        "expname": args.expname,
        "basedir": args.basedir}
    return input_dict

