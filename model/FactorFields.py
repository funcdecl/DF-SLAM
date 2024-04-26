import torch.nn
import torch.nn.functional as F
import numpy as np
import math
import skimage

from src.common import normalize_3d_coordinate, sample_pdf

def N_to_vsize(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return voxel_size
def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return torch.round((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5):
    return int(np.linalg.norm(reso)/step_ratio)

def grid_mapping(positions, freq_bands, aabb):
    #sawtooth
    aabbSize = max(aabb[1] - aabb[0])
    scale = aabbSize[..., None] / freq_bands
    pts_local = (positions - aabb[0])[..., None] % scale
    pts_local = pts_local / (scale / 2) - 1
    pts_local = pts_local.clamp(-1., 1.)
    return pts_local


def dct_dict(n_atoms_fre, size, n_selete, dim=2):
    """
    Create a dictionary using the Discrete Cosine Transform (DCT) basis. If n_atoms is
    not a perfect square, the returned dictionary will have ceil(sqrt(n_atoms))**2 atoms
    :param n_atoms:
        Number of atoms in dict
    :param size:
        Size of first patch dim
    :return:
        DCT dictionary, shape (size*size, ceil(sqrt(n_atoms))**2)
    """
    # todo flip arguments to match random_dictionary
    p = n_atoms_fre  # int(math.ceil(math.sqrt(n_atoms)))
    dct = np.zeros((p, size))

    for k in range(p):
        basis = np.cos(np.arange(size) * k * math.pi / p)
        if k > 0:
            basis = basis - np.mean(basis)

        dct[k] = basis

    kron = np.kron(dct, dct)
    if 3 == dim:
        kron = np.kron(kron, dct)

    if n_selete < kron.shape[0]:
        idx = [x[0] for x in np.array_split(np.arange(kron.shape[0]), n_selete)]
        kron = kron[idx]

    for col in range(kron.shape[0]):
        norm = np.linalg.norm(kron[col]) or 1
        kron[col] /= norm

    kron = torch.FloatTensor(kron)
    return kron


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[..., :-1]  # [N_rays, N_samples]
    return alpha, weights, T[..., -1:]


class MLPMixer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim=16,
                 num_layers=2,
                 hidden_dim=64, pe=0, with_dropout=False):
        super().__init__()

        self.with_dropout = with_dropout
        self.in_dim = in_dim + 2 * in_dim * pe
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pe = pe

        backbone = []
        for l in range(num_layers):
            if l == 0:
                layer_in_dim = self.in_dim
            else:
                layer_in_dim = self.hidden_dim

            if l == num_layers - 1:
                layer_out_dim, bias = out_dim, False
            else:
                layer_out_dim, bias = self.hidden_dim, True

            backbone.append(torch.nn.Linear(layer_in_dim, layer_out_dim, bias=bias))

        self.backbone = torch.nn.ModuleList(backbone)
        # torch.nn.init.constant_(backbone[0].weight.data, 1.0/self.in_dim)

    def forward(self, x, is_train=True):
        # x: [B, 3]
        h = x
        if self.pe > 0:
            h = torch.cat([h, positional_encoding(h, self.pe)], dim=-1)

        if self.with_dropout and is_train:
            h = F.dropout(h, p=0.1)

        for l in range(self.num_layers):
            h = self.backbone[l](h)
            if l != self.num_layers - 1:  # l!=0 and
                h = F.relu(h, inplace=True)
                # h = torch.sin(h)
        # sigma, feat = h[...,0], h[...,1:]
        #sdf = torch.tanh(h)
        return h


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, cfg, inChanel, num_layers=3, hidden_dim=64, feape=0):
        super(MLPRender_Fea, self).__init__()

        self.cfg = cfg
        self.in_mlpC = inChanel + 2 * feape * inChanel
        #self.in_mlpC = inChanel
        self.num_layers = num_layers
        self.feape = feape

        mlp = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_mlpC   #输入层
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim, bias = 3, False  # 3 rgb
            else:
                out_dim, bias = hidden_dim, True

            mlp.append(torch.nn.Linear(in_dim, out_dim, bias=bias))

        self.mlp = torch.nn.ModuleList(mlp)
        # torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, features):

        indata = [features]

        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]

        h = torch.cat(indata, dim=-1)
        for l in range(self.num_layers):
            h = self.mlp[l](h)
            if l != self.num_layers - 1:
                h = F.gelu(h)

        rgb = torch.sigmoid(h)
        return rgb

class FactorFields(torch.nn.Module):
    def __init__(self, cfg, device):
        super(FactorFields, self).__init__()

        self.cfg = cfg
        self.device = device

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.n_scene, self.scene_idx = 1, 0

        self.alphaMask = None
        self.coeff_type, self.basis_type = cfg['model']['coeff_type'], cfg['model']['basis_type']

        bbox = np.transpose(self.cfg['mapping']['bound'])
        self.setup_params(bbox)

        self.is_unbound = False
        self.n_level = 2
        self.bound = torch.FloatTensor(self.cfg['mapping']['bound']).to(self.device)

        self.coeffs_g = self.init_coef(self.cfg['model']['coeff_g_init'], self.basis_dims, self.coeff_reso)
        self.coeffs_c = self.init_coef(self.cfg['model']['coeff_c_init'], self.basis_dims, self.coeff_reso)

        self.basises_g = self.init_basis(self.basis_dims, self.basis_reso, self.in_dim, self.device)
        self.basises_c = self.init_basis(self.basis_dims, self.basis_reso, self.in_dim, self.device)

        out_dim = cfg['model']['out_dim']
        in_dim = sum(cfg['model']['basis_dims'])

        feat_c_dim = cfg['model']['feat_c_dim']     #颜色特征的维度

        self.linear_mat = MLPMixer(in_dim, out_dim,
                                   num_layers=cfg['model']['num_layers'],
                                   hidden_dim=cfg['model']['hidden_dim'],
                                   with_dropout=False).to(device)

        view_pe, fea_pe = cfg['rendering']['view_pe'], cfg['rendering']['fea_pe']
        num_layers, hidden_dim = cfg['rendering']['num_layers'], cfg['rendering']['hidden_dim']

        self.renderModule = MLPRender_Fea(cfg, inChanel=feat_c_dim,
                                          num_layers=num_layers,
                                          hidden_dim=hidden_dim,
                                          feape=fea_pe).to(device)


        self.inward_aabb = self.aabb

        # self.freq_bands = torch.FloatTensor(cfg.model.freq_bands).to(device)
        self.cur_volumeSize = N_to_reso(cfg['training']['volume_resoInit'] ** self.in_dim, self.aabb)

        print(f'=====> total parameters: {self.n_parameters() / 1024 ** 2}MB')

        self.output = cfg['data']['output']
        with open('metrics_output.txt', 'a') as file:  # 'a' mode appends to the file
            file.write(f'output: {self.output}\n')
            file.write(f'=====> total parameters: {self.n_parameters() / 1024 ** 2}MB\n')

    def setup_params(self, aabb):

        self.in_dim = len(aabb[0])
        self.aabb = torch.FloatTensor(aabb)[:, :self.in_dim].to(self.device)

        self.basis_dims = self.cfg['model']['basis_dims']
        self.coeff_reso = N_to_reso(self.cfg['model']['coeff_reso'] ** self.in_dim, self.aabb[:, :self.in_dim])[::-1]

        self.T_coeff = sum(self.cfg['model']['basis_dims']) * np.prod(self.coeff_reso)
        self.T_basis = self.cfg['model']['total_params'] - self.T_coeff
        scale = self.T_basis / sum(np.power(np.array(self.cfg['model']['basis_resos']), self.in_dim) * np.array(self.cfg['model']['basis_dims']))
        scale = np.power(scale, 1.0 / self.in_dim)
        self.basis_reso = np.round(np.array(self.cfg['model']['basis_resos']) * scale).astype('int').tolist()

        self.freq_bands_c = torch.FloatTensor(self.cfg['model']['freq_bands']).to(self.device) * (self.cfg['mapping']['scene_reso_c'] / float(max(self.basis_reso)) / max(self.cfg['model']['freq_bands']))
        #self.freq_bands_c = torch.FloatTensor(self.cfg['model']['freq_bands']).to(self.device)
        self.freq_bands_g = torch.FloatTensor(self.cfg['model']['freq_bands']).to(self.device) * (self.cfg['mapping']['scene_reso_g'] / float(max(self.basis_reso)) / max(self.cfg['model']['freq_bands']))

        self.n_scene = 1

        # print(self.coeff_reso,self.basis_reso,self.freq_bands)

    def init_coef(self, coeff_init, basis_dims, coeff_reso):
        coeffs = [coeff_init * torch.ones((1, sum(basis_dims), *coeff_reso), device=self.device)]
        coeffs = [torch.nn.Parameter(c) for c in coeffs]
        coeffs_list = torch.nn.ParameterList(coeffs)
        return coeffs_list

    def init_basis(self, basis_dims, basis_reso, in_dim, device):
        basises, coeffs, n_params_basis = [], [], 0

        for i, (basis_dim, reso) in enumerate(zip(basis_dims, basis_reso)):

            basises.append(torch.nn.Parameter(dct_dict(int(np.power(basis_dim, 1. / in_dim) + 1),
                                                       reso,
                                                       n_selete=basis_dim,
                                                       dim=in_dim).reshape([1, basis_dim] + [reso] * in_dim).to(device)))
        return torch.nn.ParameterList(basises)

    def get_coeff(self, xyz_sampled):
        N_points, dim = xyz_sampled.shape
        pts = self.normalize_coord(xyz_sampled).view([1, -1] + [1] * (dim - 1) + [dim])
        coeffs_g = F.grid_sample(self.coeffs_g[0], pts,
                               mode="bilinear",
                               align_corners=False,
                               padding_mode='border').view(-1, N_points).t()
        coeffs_c = F.grid_sample(self.coeffs_c[0], pts,
                                 mode="bilinear",
                                 align_corners=False,
                                 padding_mode='border').view(-1, N_points).t()
        return coeffs_g, coeffs_c

    def get_basis(self, x):
        N_points = x.shape[0]
        basises_g, basises_c = [], []

        freq_len_g = len(self.freq_bands_g)
        freq_len_c = len(self.freq_bands_c)
        xyz_g = grid_mapping(x, self.freq_bands_g, self.aabb[:, :self.in_dim]).view(1, *([1] * (self.in_dim - 1)), -1, self.in_dim, freq_len_g)
        xyz_c = grid_mapping(x, self.freq_bands_c, self.aabb[:, :self.in_dim]).view(1, *([1] * (self.in_dim - 1)), -1, self.in_dim, freq_len_c)

        for i in range(freq_len_g):
            basises_g.append(F.grid_sample(self.basises_g[i],
                                           xyz_g[..., i],
                                           mode="bilinear",
                                           align_corners=True).view(-1, N_points).T)
        if isinstance(basises_g, list):
            basises_g = torch.cat(basises_g, dim=-1)

        for i in range(freq_len_c):
            basises_c.append(F.grid_sample(self.basises_c[i],
                                           xyz_c[..., i], mode="bilinear",
                                           align_corners=True).view(-1, N_points).T)

        if isinstance(basises_c, list):
            basises_c = torch.cat(basises_c, dim=-1)
        return basises_g, basises_c

    @torch.no_grad()
    def normalize_basis(self):
        for basis in self.basises:
            basis.data = basis.data / torch.norm(basis.data, dim=(2, 3), keepdim=True)

    def get_coding(self, x):
        coeff_g, coeff_c = self.get_coeff(x)
        basises_g, basises_c = self.get_basis(x)
        return basises_g * coeff_g, basises_c * coeff_c


    def n_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total

    def get_optparam_groups(self, lr_small=0., lr_large=0.):
        grad_vars = [
            {'params': self.linear_mat.parameters(), 'lr': lr_small},
            {'params': self.coeffs_g.parameters(), 'lr': lr_large},
            {'params': self.coeffs_c.parameters(), 'lr': lr_large},
            {'params': self.basises_g.parameters(), 'lr': lr_large},
            {'params': self.basises_c.parameters(), 'lr': lr_large},
            {'params': self.renderModule.parameters(), 'lr': lr_small}
        ]

        return grad_vars

    def TV_loss(self, reg):
        total = 0
        for idx in range(len(self.basises)):
            total = total + reg(self.basises[idx]) * 1e-2
        return total

    def sdf2alpha(self, sdf, beta=10):
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def sample_point(self, rays_o, rays_d, gt_depth, truncation):
        n_stratified = self.cfg['rendering']['n_stratified']
        n_importance = self.cfg['rendering']['n_importance']
        n_rays = rays_o.shape[0]
        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=self.device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=self.device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=self.device)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.cfg['training']['perturb']:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0) - det_rays_o) / det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.cfg['training']['perturb']:
                    z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                feats_g, feats_c = self.get_coding(pts_uni_nor)
                feat = self.linear_mat(feats_g, is_train=True)
                sdf_uni = feat[..., 0]
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni)

                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=self.device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=self.device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]

        return pts, z_vals


    def normalize_coord(self, xyz_sampled):
        invaabbSize = 2.0 / (self.aabb[1] - self.aabb[0])
        return (xyz_sampled - self.aabb[0]) * invaabbSize - 1

    def basis2density(self, density_features):
        if self.cfg['rendering']['fea2denseAct'] == "softplus":
            return F.softplus(density_features + self.cfg['rendering']['density_shift'])
        elif self.cfg['rendering']['fea2denseAct'] == "relu":
            return F.relu(density_features + self.cfg['rendering']['density_shift'])


    def sdf2weights(self, sdf, z_vals):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / self.cfg['training']['trunc']) * torch.sigmoid(-sdf / self.cfg['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + self.cfg['data']['sc_factor'] * self.cfg['training']['trunc'],
                           torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)


    def get_normal(self, pts, sdf):
        d_output = torch.zeros(sdf.shape, device=self.device)
        normal = torch.autograd.grad(sdf, pts, grad_outputs=d_output, create_graph=False, retain_graph=False)[0]
        normal = normal.reshape(-1, 3)
        return normal

    def run_network(self, rays_chunk, gt_depth=None):

        n_rays_chunk = rays_chunk.shape[0]
        # sample points
        viewdirs = rays_chunk[:, 3:6]

        xyz_sampled, z_vals = self.sample_point(rays_chunk[:, :3], viewdirs, gt_depth, self.cfg['model']['truncation'])

        xyz_sampled = xyz_sampled.reshape(-1, 3)

        feats_g, feats_c = self.get_coding(xyz_sampled)

        sdf = self.linear_mat(feats_g)

        sdf = sdf.reshape(n_rays_chunk, -1)


        alpha = self.sdf2alpha(sdf)

        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device)
                                                      , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        weights_c = weights if self.cfg['mapping']['weight_c_requires_grad'] else weights.detach().clone()

        feats_c = feats_c.reshape(n_rays_chunk, -1, feats_c.shape[-1])

        feature_ray = torch.sum(weights_c[..., None] * feats_c, -2)
        rendered_rgb = self.renderModule(feature_ray)


        #raw = torch.cat([rgbs, sdf], -1)
        #raw = raw.reshape(n_rays_chunk, -1, raw.shape[-1])
        z_vals = z_vals.reshape(n_rays_chunk, -1)

        #rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)

        return rendered_rgb, rendered_depth, sdf, z_vals, alpha
