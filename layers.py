import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
    

def random_image_mask(img, filter_size):
    '''
    :param img: [B x 3 x H x W]
    :param crop_size:
    :return:
    '''
    fh, fw = filter_size
    _, _, h, w = img.size()

    if fh == h and fw == w:
        return img, None

    x = np.random.randint(0, w - fw)
    y = np.random.randint(0, h - fh)
    filter_mask = torch.ones_like(img)    # B x 3 x H x W
    filter_mask[:, :, y:y+fh, x:x+fw] = 0.0    # B x 3 x H x W
    img = img * filter_mask    # B x 3 x H x W
    return img, filter_mask


def schedule_depth_rangev3(prior_depth, ndepth, scale_facs, type='inverse'):
    with torch.no_grad():
        B,_,H,W = prior_depth.shape
        depth_center = prior_depth
        scale_fac = scale_facs.unsqueeze(0).unsqueeze(0)
        scheduled_min_depth = depth_center/(1+scale_fac)
        #scheduled_min_depth = depth_center*(1-scale_fac)
        scheduled_max_depth = depth_center*(1+scale_fac)

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype, requires_grad=False).reshape(1,-1,1,1)  / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1/scheduled_max_depth + (1/scheduled_min_depth - 1/scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype, requires_grad=False).reshape(1,-1,1,1)  / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth-1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B,1,H,W).to(scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def schedule_depth_rangev2(prior_depth, ndepth, scale_fac, type='inverse'):
    with torch.no_grad():
        B,_,H,W = prior_depth.shape
        depth_center = prior_depth

        scheduled_min_depth = depth_center/(1+scale_fac)
        #scheduled_min_depth = depth_center*(1-scale_fac)
        scheduled_max_depth = depth_center*(1+scale_fac)

        if type == 'inverse':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype, requires_grad=False).reshape(1,-1,1,1)  / (ndepth - 1)  # 1 D H W
            inverse_depth_hypo = 1/scheduled_max_depth + (1/scheduled_min_depth - 1/scheduled_max_depth) * itv
            depth_range = 1 / inverse_depth_hypo

        elif type == 'linear':
            itv = torch.arange(0, ndepth, device=prior_depth.device, dtype=prior_depth.dtype, requires_grad=False).reshape(1,-1,1,1)  / (ndepth - 1)  # 1 D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        elif type == 'log':
            itv = []
            for K in range(ndepth):
                K_ = torch.FloatTensor([K])
                itv.append(torch.exp(torch.log(torch.FloatTensor([0.1])) + torch.log(torch.FloatTensor([1 / 0.1])) * K_ / (ndepth-1)))
            itv = torch.FloatTensor(itv).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(B,1,H,W).to(scheduled_min_depth.device)  # B D H W
            depth_range = scheduled_min_depth + (scheduled_max_depth - scheduled_min_depth) * itv

        else:
            raise NotImplementedError

    return depth_range  # (B.D,H,W)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)  # B 3 3
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def transformation_from_parameters_v2(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle).reshape(-1, 1, 1, 4, 4)  # B 1, 1, 4 4
    t = translation.clone()

    if invert:
        R = R.transpose(3, 4)
        t *= -1

    T = get_translation_matrix_v2(t)  # B H W 4 4

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M  # B H W 4 4

def get_translation_matrix_v2(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    B, H, W, _ = translation_vector.shape
    T = torch.zeros(B, H, W, 4, 4).to(device=translation_vector.device)
    T[..., 0, 0] = 1
    T[..., 1, 1] = 1
    T[..., 2, 2] = 1
    T[..., 3, 3] = 1
    T[..., :3, 3] = translation_vector

    return T  # B H W 4 4


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlock1x1(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1x1, self).__init__()

        self.conv = Conv1x1(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv1x1(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out

class Conv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", group_channel=8, **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return
    
def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return



class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D_S(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # K： B 4 4
        # T: B H W 4 4
        if len(T.shape) == 5:
            B,H,W,_,_ = T.shape
            K = K[:,None,None]
            points = points.reshape(-1,4,H,W,1).permute(0,2,3,1,4)  # B H W 4 1
        P = torch.matmul(K, T)[..., :3, :]  # B H W 3 4

        cam_points = torch.matmul(P, points)  # B H W 3 1

        pix_coords = cam_points[..., :2, :] / (cam_points[..., 2:3, :] + self.eps)  # [B H W] 2 1
        if len(T.shape) == 5:
            pix_coords = pix_coords[..., 0]  # B H W 2
        else:
            pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
            pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

def updown_sample(x, scale_fac):
    """Upsample input tensor by a factor of scale_fac
    """
    return F.interpolate(x, scale_factor=scale_fac, mode="nearest")

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class MVS_SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(MVS_SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        # print('mask: {}'.format(mask.shape))
        # print('x: {}'.format(x.shape))
        # print('y: {}'.format(y.shape))
        # x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        # y = y.permute(0, 3, 1, 2)
        # mask = mask.permute(0, 3, 1, 2)

        # x = self.refl(x)
        # y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask.float())
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output, SSIM_mask
        # return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def generate_costvol_v1(ref, src, K, invK, depth_priors, pose, num_depth_bins, num_groups, backprojector, projector):
    cost_vols = []
    _, _, H, W = depth_priors.shape
    for batch_idx in range(len(ref)):
        ref_feat = ref[batch_idx:batch_idx + 1]
        source_feat = src[batch_idx:batch_idx + 1]
        source_feat = source_feat.repeat([num_depth_bins, 1, 1, 1])
        with torch.no_grad():
            _K = K[batch_idx:batch_idx + 1]
            _invK = invK[batch_idx:batch_idx + 1]
            depth_prior = depth_priors[batch_idx:batch_idx + 1]
            _lookup_poses = pose[batch_idx:batch_idx + 1, 0]
            world_points = backprojector(depth_prior, _invK)
            pix_locs = projector(world_points, _K, _lookup_poses).squeeze(1)
        warped = F.grid_sample(source_feat, pix_locs, padding_mode='zeros', mode='bilinear', align_corners=True)
        cost = warped * ref_feat
        
        cost = cost.reshape(num_depth_bins, -1, num_groups, H, W).mean(1)
        cost_vols.append(cost)  # D C H W
    cost_vols = torch.stack(cost_vols, 0)  # B D C H W
    return cost_vols

def localmax_v1(cost_prob, radius, casbin, prior_depth):
    pred_idx = torch.argmax(cost_prob, 1, keepdim=True)  # B 1 H W
    pred_idx_low = pred_idx - radius
    pred_idx = torch.arange(0, 2*radius+1, 1, device=pred_idx.device).reshape(1, 2*radius+1,1,1)
    pred_idx = pred_idx + pred_idx_low  # B M H W
    pred_idx = torch.clamp(pred_idx, 0, casbin-1)
    cost_prob_ = torch.gather(cost_prob, 1, pred_idx)
    prior_depth_ = torch.gather(prior_depth, 1, pred_idx)
    depth_mvs = torch.sum(cost_prob_*prior_depth_, 1)/torch.sum(cost_prob_, 1)    
    
    return depth_mvs


