import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *


DEBUG = False


## Positional encoding (section 5.1)
class Embedder:
    def __init__(self, args):
        self.embed_kwargs = {
            'include_input' : True,
            'input_dims' : 3,
            'max_freq_log2' : args.multires-1,
            'num_freqs' : args.multires,
            'log_sampling' : True,
            'periodic_fns' : [torch.sin, torch.cos],}
        self.kwargs = self.embed_kwargs
        self.create_embedding_fn()

        
    def create_embedding_fn(self):
        embed_fns = []
        d         = self.kwargs['input_dims']
        out_dim   = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs  = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim   = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


## Model
class BasicNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(BasicNeRF, self).__init__()
        self.D              = D
        self.W              = W
        self.input_ch       = input_ch
        self.input_ch_views = input_ch_views
        self.skips          = skips
        self.use_viewdirs   = use_viewdirs
        
        self.pts_linears    = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/BasicNeRF/blob/master/run_BasicNeRF_helpers.py#L104-L105)
        self.views_linears  = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear   = nn.Linear(W, 1)
            self.rgb_linear     = nn.Linear(W//2, 3)
        else:
            self.output_linear  = nn.Linear(W, output_ch)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h                      = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha   = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h       = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb     = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    



class NeuralRendering(nn.Module):
    def __init__(self, args, device):
        super(NeuralRendering, self).__init__()
        ## 포지셔널 인코딩 
        self.args     = args
        self.device   = device

        self.embedder = Embedder(args)
        if args.use_viewdirs:
            self.embeder_dirs = Embedder(args)


        ## BasicNeRF 베이스 모델
        skips = [4]
        if args.N_importance > 0:
            output_ch = 5
        else:
            output_ch = 4

        self.model = BasicNeRF(
            D = args.netdepth, W = args.netwidth, 
            input_ch = self.embedder.out_dim, output_ch = output_ch, skips = skips,
            input_ch_views = self.embeder_dirs.out_dim, use_viewdirs = args.use_viewdirs).to(device)

        self.model_fine = None
        if args.N_importance > 0:
            self.model_fine = BasicNeRF(
                D = args.netdepth_fine, W = args.netwidth_fine, 
                input_ch = self.embedder.out_dim, output_ch = output_ch, skips = skips,
                input_ch_views = self.embeder_dirs.out_dim, use_viewdirs = args.use_viewdirs).to(device)


        ## Load checkpoints
        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(args.basedir, args.expname, f) for \
            f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if 'tar' in f]

        print('>>> Founding ckpts ...', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('>>> Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)

            ## Load model
            self.model.load_state_dict(ckpt['model_state_dict'])
            if self.model_fine is not None:
                self.model_fine.load_state_dict(ckpt['model_fine_state_dict'])


    def compute_outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
        """T
        ransforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[..., None, :], dim = -1)

        rgb   = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std


        alpha     = 1.-torch.exp(-F.relu(raw[...,3] + noise) * dists)  # [N_rays, N_samples]
        weights   = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map   = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map  = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map   = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])
        return rgb_map, disp_map, acc_map, weights, depth_map


    def batch_rays_rendering(
        self, 
        ray_batch,
        N_samples,
        N_importance = 0,
        perturb = 0.,
        raw_noise_std = 0,
        retraw = False,
        lindisp = False,
        white_bkgd = False):
        """
        Volumetric rendering.
        Args:
            ray_batch: array of shape [batch_size, ...]. All information necessary
                    for sampling along a ray, including: ray origin, ray direction, min
                    dist, max dist, and unit-magnitude viewing direction.
            network_fn:       function. Model for predicting RGB and density at each point in space.
            network_query_fn: function used for passing queries to network_fn.

            N_samples: int. Number of different times to sample along each ray.
            retraw: bool. If True, include model's raw, unprocessed predictions.
            lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
            perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.
            N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
            network_fine: "fine" network with same spec as network_fn.
            white_bkgd: bool. If True, assume a white background.
            raw_noise_std: ...
            verbose: bool. If True, print more debugging info.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
            disp_map: [num_rays]. Disparity map. 1 / depth.
            acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
            raw: [num_rays, num_samples, 4]. Raw predictions from model.
            rgb0: See rgb_map. Output for coarse model.
            disp0: See disp_map. Output for coarse model.
            acc0: See acc_map. Output for coarse model.
            z_std: [num_rays]. Standard deviation of distances along ray for each sample.
        """
        def batchify(fn, netchunk):
            """
            Constructs a version of 'fn' that applies to smaller batches.
            """
            def ret(inputs):
                return torch.cat([fn(inputs[i:i+netchunk]) for i in range(0, inputs.shape[0], netchunk)], 0)
                
            if netchunk is None:
                return fn
            return ret


        N_rays         = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
        viewdirs       = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds         = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far      = bounds[...,0], bounds[...,1] # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids   = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper  = torch.cat([mids, z_vals[...,-1:]], -1)
            lower  = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)
            z_vals = lower + (upper - lower) * t_rand

        ## coarse model forward
        pts      = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] ## [N_rays, N_samples, 3]
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        embedded = self.embedder.embed(pts_flat)

        if viewdirs is not None:
            input_dirs    = viewdirs[:, None].expand(pts.shape)
            input_flat    = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeder_dirs.embed(input_flat)
            embedded      = torch.cat([embedded, embedded_dirs], dim = -1)

        outputs_flat = batchify(self.model, self.args.netchunk)(embedded)
        outputs_raw  = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
        rgb_map, disp_map, acc_map, weights, _ = self.compute_outputs(outputs_raw, z_vals, rays_d, raw_noise_std, white_bkgd)


        if N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples  = compute_sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.))
            z_samples  = z_samples.detach()
            z_vals, _  = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts        = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            if self.model_fine is None:
                net_fine = self.model
            else:
                net_fine = self.model_fine

            ## fine model forward
            pts        = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] ## [N_rays, N_samples, 3]
            pts_flat   = torch.reshape(pts, [-1, pts.shape[-1]])
            embedded   = self.embedder.embed(pts_flat)

            if viewdirs is not None:
                input_dirs    = viewdirs[:, None].expand(pts.shape)
                input_flat    = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
                embedded_dirs = self.embeder_dirs.embed(input_flat)
                embedded      = torch.cat([embedded, embedded_dirs], dim = -1)

            outputs_flat = batchify(net_fine, self.args.netchunk)(embedded)
            outputs_raw  = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
            rgb_map, disp_map, acc_map, weights, _ = self.compute_outputs(outputs_raw, z_vals, rays_d, raw_noise_std, white_bkgd)


        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
        if retraw:
            ret['raw'] = outputs_raw
        if N_importance > 0:
            ret['rgb0']  = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0']  = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")
        return ret


    def batch_rays_processing(self, rays_flat, chunk = 1024*32, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.batch_rays_rendering(rays_flat[i:i+chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    def forward(
        self, H, W, K, chunk = 1024 * 32, rays = None, ndc = True, 
        near = 0.0, far = 1.0, c2w = None, c2w_staticcam = None, use_viewdirs = False, **kwargs):
        if c2w is not None:
            ## special case to render full image
            rays_o, rays_d = compute_rays(H, W, K, c2w, "torch")
        else:
            ## use provided ray batch
            rays_o, rays_d = rays

        ## provide ray directions as input
        if use_viewdirs:
            viewdirs = rays_d
            if c2w_staticcam is not None: ## special case to visualize effect of viewdirs
                rays_o, rays_d = compute_rays(H, W, K, c2w_staticcam, "torch")
            viewdirs = viewdirs / torch.norm(viewdirs, dim = -1, keepdim = True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape # [..., 3]
        if ndc: ## for forward facing scenes
            rays_o, rays_d = compute_ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        ## Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        near   = near * torch.ones_like(rays_d[...,:1])
        far    = far * torch.ones_like(rays_d[...,:1])
        rays   = torch.cat([rays_o, rays_d, near, far], -1)

        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        all_ret = self.batch_rays_processing(rays, chunk, **kwargs)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list  = [all_ret[k] for k in k_extract]
        ret_dict  = {k : all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]