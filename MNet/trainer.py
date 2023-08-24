import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
import time
import random
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import json

from utils import readlines, sec_to_hm_str, seed_worker, get_dist_info, DistributedSampler
from layers import SSIM, BackprojectDepth, Project3D, transformation_from_parameters, \
    disp_to_depth, get_smooth_loss, compute_depth_errors, schedule_depth_rangev2, \
    generate_costvol_v1, localmax_v1, random_image_mask, schedule_depth_rangev3
import datasets
from MNet import networks
import matplotlib.pyplot as plt

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.mnet_parameters_to_train = []

        self.train_mnet = True
        self.train_snet_and_pose = True

        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        if self.opt.ddp:
            dist.init_process_group(backend='nccl', )
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        assert len(self.opt.frame_ids) > 1, "frame_ids must have more than 1 frame specified"

        # check the frames we need the dataloader to load
        frames_to_load = self.opt.frame_ids.copy()
        self.matching_ids = self.opt.matching_ids
        if self.local_rank == 0:
            print('Loading frames: {}'.format(frames_to_load))

        # MODEL SETUP
        # single-frame encoder
        if self.opt.snet=="resnet":
            self.models["s_encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        elif self.opt.snet=="hrnet":
            self.models["s_encoder"] = networks.hrnet18(self.opt.weights_init == "pretrained")
        if self.opt.ddp:
            self.models["s_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["s_encoder"])
        self.models["s_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["s_encoder"].parameters())
        
        # single-frame depth decoder
        if self.opt.snet=="resnet":
            self.models["s_depth"] = networks.DepthDecoder(self.models["s_encoder"].num_ch_enc, self.opt.scales)
        elif self.opt.snet=="hrnet":
            self.models["s_depth"] = networks.DepthDecoder_MSF(
                self.models["s_encoder"].num_ch_enc, self.opt.scales)
        if self.opt.ddp:
            self.models["s_depth"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["s_depth"])
        self.models["s_depth"].to(self.device)
        self.parameters_to_train += list(self.models["s_depth"].parameters())

        # load pose is worse than joint training!
        if not self.opt.load_pose:
            # pose encoder
            self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=self.num_pose_frames)
            if self.opt.ddp:
                self.models["pose_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose_encoder"])
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            
            # pose decoder
            self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
            if self.opt.ddp:
                self.models["pose"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose"])
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())


        # multi-frame depth encoder
        self.models["m_encoder"] = networks.FPN4(base_channels=8, scale=self.opt.prior_scale)
        if self.opt.ddp:
            self.models["m_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["m_encoder"])
        self.models["m_encoder"].to(self.device)
        self.mnet_parameters_to_train += list(self.models["m_encoder"].parameters())

        # warping layer
        self.backprojector = BackprojectDepth(batch_size=self.opt.num_depth_bins, height=self.opt.height//2**(self.opt.prior_scale), width=self.opt.width//2**(self.opt.prior_scale))
        self.projector = Project3D(batch_size=self.opt.num_depth_bins, height=self.opt.height//2**(self.opt.prior_scale), width=self.opt.width//2**(self.opt.prior_scale))
        self.backprojector.to(self.device)
        self.projector.to(self.device)
        
        
        # 3d unet for regularization
        if self.opt.num_depth_bins >=8:
            self.models["reg3d"] = networks.reg3d(in_channels=self.opt.reg3d_c, base_channels=self.opt.reg3d_c, down_size=3)  # FIXME hardcoded
        else:
            self.models["reg3d"] = networks.reg2d(input_channel=self.opt.reg3d_c, base_channel=self.opt.reg3d_c)  # FIXME hardcoded
        if self.opt.ddp:
            self.models["reg3d"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["reg3d"])
        self.models["reg3d"].to(self.device)
        self.mnet_parameters_to_train += list(self.models["reg3d"].parameters())

        # upsample
        if self.opt.convex_up:
            self.models["up"] = networks.convex_upsample_layer(feature_dim=8*2**self.opt.prior_scale, scale=self.opt.prior_scale)
            if self.opt.ddp:
                self.models["up"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["up"])
            self.models["up"].to(self.device)
            self.mnet_parameters_to_train += list(self.models["up"].parameters())


        if self.opt.ddp:
            for key in self.models.keys():
                self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        
        self.model_optimizer = optim.Adam([
            {'params': self.parameters_to_train, 'lr': self.opt.learning_rate},
            {'params': self.mnet_parameters_to_train, 'lr': self.opt.learning_rate*self.opt.lr_fac},
        ])
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.local_rank == 0:
            print("Training model named:\n  ", self.opt.model_name)
            print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
            print("Training is using:\n  ", self.device)

        # DATA
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        if self.opt.dataset.startswith('kitti'):
            fpath = "./splits/{}"+"/{}_files.txt"
            train_filenames = readlines(fpath.format(self.opt.split, "train"))
            val_filenames = readlines(fpath.format(self.opt.split, "val"))
        else:
            assert False, "Not implemented yet"
        img_ext = '.png' if self.opt.png else '.jpg'

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=True, img_ext=img_ext, load_pose=self.opt.load_pose)

        if self.opt.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,
                worker_init_fn=seed_worker)

        self.num_total_steps = len(self.train_loader) * self.opt.num_epochs

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            frames_to_load, 4, is_train=False, img_ext=img_ext, load_pose=self.opt.load_pose)

        if self.opt.ddp:
            rank, world_size = get_dist_info()
            self.world_size = world_size
            val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, shuffle=False,
                num_workers=4, pin_memory=True, drop_last=False, sampler=val_sampler)
        else:
            self.world_size = 1
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.val_iter = iter(self.val_loader)
        self.writers = {}
        for mode in ["train", "val"]:
            if self.local_rank == 0:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if self.local_rank == 0:
            print("Using split:\n  ", self.opt.split)
            print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), len(val_dataset)))
        if self.opt.ddp:
            self.opt.log_frequency = self.opt.log_frequency//self.world_size
        self.save_opts()

        if self.opt.adaptive_range:
            self.depth_bin_facs = torch.full(size=(self.opt.height // (2 ** self.opt.prior_scale), self.opt.width // (2 ** self.opt.prior_scale)), fill_value=self.opt.depth_bin_fac).to(self.device)

    def set_train(self):
        """Convert all models to training mode
        """
        for k, m in self.models.items():
            if (self.train_mnet and k in ["m_encoder", "reg3d", "up"] ) or \
               (self.train_snet_and_pose and k in ["pose_encoder", "pose", "s_depth", "s_encoder"]):
                m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.save_model()
        for self.epoch in range(self.opt.num_epochs):
            if self.opt.ddp:
                self.train_loader.sampler.set_epoch(self.epoch)
            
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.epoch>=self.opt.start_save_epochs:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        if self.local_rank == 0:
            print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            
            before_op_time = time.time()
            
            
            outputs, losses = self.process_batch(inputs, is_train=True)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                if self.local_rank == 0:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                if self.local_rank == 0:
                    self.log("train", inputs, outputs, losses)
                self.val()

            if self.opt.save_intermediate_models and late_phase:
                self.save_model(save_step=True)

            self.step += 1
        self.model_lr_scheduler.step()    

    def process_batch(self, inputs, is_train=False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        if not self.opt.load_pose:
            if not self.train_snet_and_pose:
                with torch.no_grad():
                    pose_pred = self.predict_poses(inputs, None)
            else:
                pose_pred = self.predict_poses(inputs, None)
            outputs.update(pose_pred)

        else:
            for f_i in self.opt.frame_ids[1:]:
                outputs[("cam_T_cam", 0, f_i)] = inputs['relative_pose', f_i]

        # grab poses + frames and stack for input to the multi frame network
        relative_poses = [inputs[('relative_pose', idx)] for idx in self.matching_ids[1:]]
        relative_poses = torch.stack(relative_poses, 1)

        # multi-frame feature extraction
        ref_match_feat, ref_context_feat = self.models["m_encoder"](inputs["color_aug", 0, 0])
        src_match_feats = []
        for f_i in self.matching_ids[1:]:
            src_match_feat, _ = self.models["m_encoder"](inputs["color_aug", f_i, 0])
            src_match_feats.append(src_match_feat)

        # single frame path
        if self.train_snet_and_pose:
            feats = self.models["s_encoder"](inputs["color_aug", 0, 0])
            outputs.update(self.models['s_depth'](feats))
        else:
            with torch.no_grad():
                feats = self.models["s_encoder"](inputs["color_aug", 0, 0])
                outputs.update(self.models['s_depth'](feats))

        
        

        # the single-frame depth prior for mnet
        disp_prior = F.interpolate(outputs[("disp", 0)], [self.opt.height//(2**self.opt.prior_scale), self.opt.width//(2**self.opt.prior_scale)], mode="nearest").clone().detach()
        disp_scaled = 1/self.opt.max_depth + disp_prior * (1/self.opt.min_depth - 1/self.opt.max_depth)
        depth_prior = 1 / disp_scaled
        if self.opt.adaptive_range:
            s_depth_prior = schedule_depth_rangev3(depth_prior,
                                        ndepth=self.opt.num_depth_bins, 
                                        scale_facs=self.depth_bin_facs.clone().detach(),
                                        type=self.opt.schedule_type)    # (B,D,H,W)
        else:
            s_depth_prior = schedule_depth_rangev2(depth_prior,
                                        ndepth=self.opt.num_depth_bins, 
                                        scale_fac=self.opt.depth_bin_fac,
                                        type=self.opt.schedule_type)    # (B,D,H,W)
            
        

        # mnet cost volume
        cor_feats = generate_costvol_v1(ref_match_feat, src_match_feats[0],
                                    inputs[('K', 2)], inputs[('inv_K', 2)],
                                    s_depth_prior, relative_poses[:, 0:1],
                                    self.opt.num_depth_bins,
                                    self.opt.reg3d_c,
                                    self.backprojector,
                                    self.projector)


        cost_prob = self.models["reg3d"](cor_feats)  # B D H W
        cost_prob = F.softmax(cost_prob, 1)
        
        depth_m = localmax_v1(cost_prob, self.opt.norm_radius, self.opt.num_depth_bins, s_depth_prior)  # B H W

        
        # mask aug depth prediction        
        ori_H, ori_W = inputs["color_aug", 0, 0].shape[2:]
        masked_img, this_aug_mask = random_image_mask(inputs["color_aug", 0, 0], [ori_H//3, ori_W//3])
        ref_aug_feat, _ = self.models["m_encoder"](masked_img)
        cor_feats = generate_costvol_v1(ref_aug_feat, src_match_feats[0],
                                    inputs[('K', 2)], inputs[('inv_K', 2)],
                                    s_depth_prior, relative_poses[:, 0:1],
                                    self.opt.num_depth_bins,
                                    self.opt.reg3d_c,
                                    self.backprojector,
                                    self.projector)

        # norm depth index
        cost_prob = self.models["reg3d"](cor_feats)  # B D H W
        cost_prob = F.softmax(cost_prob, 1)
        
        depth_m_aug = localmax_v1(cost_prob, self.opt.norm_radius, self.opt.num_depth_bins, s_depth_prior)  # B H W

        this_mask = F.interpolate(this_aug_mask, [depth_m_aug.shape[1], depth_m_aug.shape[2]], mode="bilinear", align_corners=True).sum(1).to(torch.bool)
        #this_masked_loss = F.smooth_l1_loss(depth_m_aug[this_mask], depth_m[this_mask], size_average=True) * self.opt.mask_lw
        this_masked_loss = F.smooth_l1_loss(depth_m_aug[this_mask], depth_m[this_mask], reduction='mean') * self.opt.mask_lw
        
        outputs["masked_depth"] = depth_m_aug
        outputs["masked_aug"] = this_aug_mask

        if not self.opt.convex_up:
            depth_m = F.interpolate(depth_m.unsqueeze(1), [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)[:, 0]
        else:
            depth_m = self.models["up"](depth_m, ref_context_feat)
        outputs['depth_m'] = depth_m
        
        if self.opt.adaptive_range:
            depth_prior = depth_prior.squeeze(1).detach()
            depth_m = F.interpolate(depth_m.detach().unsqueeze(1), (self.opt.height // (2 ** self.opt.prior_scale), self.opt.width // (2 ** self.opt.prior_scale)), mode="bilinear", align_corners=True)[:, 0]
            delta, _= torch.max(torch.cat((depth_m/depth_prior, depth_prior/depth_m), 0), 0)
            #print(delta.shape)
            delta = (delta - 1)* self.opt.delta_fac
            #delta = torch.clip(delta, min=self.opt.delta_min, max=self.opt.delta_max)
            self.depth_bin_facs = self.depth_bin_facs * 0.99 + delta * 0.01
        
        # single_frame reprojection loss
        self.generate_images_pred(inputs, outputs)
        single_frame_losses = self.compute_losses(inputs, outputs)

        self.generate_images_pred(inputs, outputs, is_muti_frame=True)
        multi_frame_losses = self.compute_losses(inputs, outputs, is_muti_frame=True)
        
        multi_frame_losses["masked_loss"] = this_masked_loss * self.opt.mask_lw
        multi_frame_losses["loss"] += multi_frame_losses["masked_loss"]
        

        if self.train_snet_and_pose:
            for key, val in single_frame_losses.items():
                if key in multi_frame_losses.keys():
                    multi_frame_losses[key] += val
                else:
                    multi_frame_losses[key] = val

        return outputs, multi_frame_losses

    def predict_poses(self, inputs, features=None):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
        for f_i in self.opt.frame_ids[1:]:
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        for fi in self.matching_ids[1:]:
            inputs[('relative_pose', fi)] = outputs[("cam_T_cam", 0, fi)].clone().detach()

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            if self.local_rank == 0:
                self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs, is_muti_frame=False):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        if is_muti_frame:
            depth_m = outputs["depth_m"]
            for _, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                T = T.detach()
                source_scale = 0
                cam_points = self.backproject_depth[source_scale](depth_m, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                outputs[("m_mask", frame_id)] = ((pix_coords<-1) | (pix_coords>1)).sum(-1) > 0  # 1 H W
                outputs[("m_color", frame_id)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border", align_corners=True)
            return

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target, ssim_lw=None):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            if ssim_lw is None:
                reprojection_loss = self.opt.ssim_lw * ssim_loss + (1-self.opt.ssim_lw) * l1_loss
            else:
                reprojection_loss = ssim_lw * ssim_loss + (1-ssim_lw) * l1_loss
        
        return reprojection_loss

    @staticmethod
    def compute_loss_masks(reprojection_loss, identity_reprojection_loss):
        """ Compute loss masks for each of standard reprojection and depth hint
        reprojection"""

        if identity_reprojection_loss is None:
            # we are not using automasking - standard reprojection loss applied to all pixels
            reprojection_loss_mask = torch.ones_like(reprojection_loss)

        else:
            # we are using automasking
            all_losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)
            idxs = torch.argmin(all_losses, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs == 0).float()

        return reprojection_loss_mask

    def compute_losses(self, inputs, outputs, is_muti_frame=False):
        """Compute the reprojection, smoothness and proxy supervised losses for a minibatch
        """
        losses = {}
        total_loss = 0
        reprojection_losses = []

        
        if is_muti_frame:
            scale = 0
            loss = 0
            source_scale = 0
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("m_color", frame_id)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # B N H W

            
            reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)  # B 1 H W
            reprojection_loss_mask = torch.ones_like(reprojection_loss)
            outputs["m_reprojection_loss"] = reprojection_loss


            # multi_frame depth mask loss
            outputs["reprojection_loss_mask"] = reprojection_loss_mask

            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)
            outputs['m_reproj_loss'] = reprojection_loss
            #print("m", reprojection_loss, "\n\n")
            loss += reprojection_loss

            # smooth loss
            disp_m = (1/outputs["depth_m"].unsqueeze(1) - 1/self.opt.max_depth) / (1/self.opt.min_depth - 1/self.opt.max_depth)
            mean_disp = disp_m.mean(2, True).mean(3, True)
            norm_disp = disp_m / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
                
            losses['m_smooth_loss/{}'.format(scale)] = smooth_loss
            loss += self.opt.multi_frame_disparity_smoothness * smooth_loss
            losses["loss"] = loss
            return losses

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            
            

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)  # B 2 H W

            # auto mask
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
                reprojection_loss_mask = self.compute_loss_masks(reprojection_loss, identity_reprojection_loss)
                
            else:
                reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
                reprojection_loss_mask = torch.ones_like(reprojection_loss)
            
            if scale == 0:
                outputs['s_reproj_loss'] = reprojection_loss

            # standard reprojection loss
            reprojection_loss = reprojection_loss * reprojection_loss_mask
            reprojection_loss = reprojection_loss.sum() / (reprojection_loss_mask.sum() + 1e-7)
            #print("s_reprojection_loss", reprojection_loss)
            
            loss += reprojection_loss
            
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            losses['s_smooth_loss/{}'.format(scale)] = smooth_loss

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses
    
    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        min_depth = 1e-3
        max_depth = 80

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > min_depth) * (depth_gt < max_depth)

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        if self.local_rank == 0:
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s = 0  # log only max scale
            for frame_id in self.opt.frame_ids:
                writer.add_image("color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                if s == 0 and frame_id != 0:
                    writer.add_image("color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)

            disp = colormap(outputs[('disp', s)][j, 0])
            writer.add_image( "disp_single/{}".format(j), disp, self.step)

            disp_m = 1 / outputs["depth_m"]
            disp_m = colormap(disp_m[j])
            writer.add_image( "disp_m/{}".format(j), disp_m, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_step=False):
        """Save model weights to disk
        """
        if self.local_rank == 0:
            if save_step:
                save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch, self.step))
            else:
                save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
            
            if self.epoch == self.opt.num_epochs-1:
                save_folder = os.path.join(self.log_path, "models", "last")

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                if self.opt.ddp:
                    to_save = model.module.state_dict()
                else:
                    to_save = model.state_dict()
                if model_name == "reg3d" and self.opt.adaptive_range:
                    to_save['depth_bin_facs'] = self.depth_bin_facs
                torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)

   
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis
