# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import SNet.networks as networks

import MNet.networks as m_networks
from layers import generate_costvol_v1, localmax_v1, schedule_depth_rangev2, schedule_depth_rangev3

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        if self.opt.pytorch_random_seed is not None:
            torch.manual_seed(self.opt.pytorch_random_seed)
            torch.cuda.manual_seed(self.opt.pytorch_random_seed)
            torch.cuda.manual_seed_all(self.opt.pytorch_random_seed)
            np.random.seed(self.opt.pytorch_random_seed)
            random.seed(self.opt.pytorch_random_seed)
            #torch.backends.cudnn.deterministic = True
            #torch.backends.cudnn.benchmark = False

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.frame_ids == [0])

        if self.opt.snet=="resnet":
            self.models["s_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
        elif self.opt.snet=="hrnet":
            self.models["s_encoder"] = m_networks.hrnet18(self.opt.weights_init == "pretrained")
        self.models["s_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["s_encoder"].parameters())

        if self.opt.snet=="resnet":
            self.models["s_depth"] = networks.DepthDecoder(
                self.models["s_encoder"].num_ch_enc, self.opt.scales)
        elif self.opt.snet=="hrnet":
            self.models["s_depth"] = m_networks.DepthDecoder_MSF(
                self.models["s_encoder"].num_ch_enc, self.opt.scales)
        self.models["s_depth"].to(self.device)
        self.parameters_to_train += list(self.models["s_depth"].parameters())

        if self.use_pose_net:
            
            self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        #train_filenames = readlines(fpath.format("test"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        
        self.writers = {}
        for mode in ["train", "val"]:
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

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if self.opt.use_many_frame_depth:
            self.m_models = {}
            # SNet encoder
            s_encoder_path = os.path.join(self.opt.teacher_weights_folder, "s_encoder.pth")
            s_encoder_dict = torch.load(s_encoder_path)
            if self.opt.snet=="resnet":
                self.m_models["s_encoder"] = m_networks.ResnetEncoder(num_layers=18, pretrained=False)
            elif self.opt.snet=="hrnet":
                self.m_models["s_encoder"] = m_networks.hrnet18(False)
            try:
                self.m_models["s_encoder"] .load_state_dict(s_encoder_dict, strict=True)
            except:
                model_dict = self.m_models["s_encoder"].state_dict()
                self.m_models["s_encoder"].load_state_dict({k: v for k, v in s_encoder_dict.items() if k in model_dict})
            self.m_models["s_encoder"] .eval()
            self.m_models["s_encoder"] .cuda()

            # SNet decoder
            s_decoder_path = os.path.join(self.opt.teacher_weights_folder, "s_depth.pth")
            s_depth_decoder_dict = torch.load(s_decoder_path)
            if self.opt.snet=="resnet":
                self.m_models["s_depth_decoder"] = m_networks.DepthDecoder(self.m_models["s_encoder"].num_ch_enc)
            elif self.opt.snet=="hrnet":
                self.m_models["s_depth_decoder"] = m_networks.DepthDecoder_MSF(self.m_models["s_encoder"].num_ch_enc, [0], num_output_channels=1)
            self.m_models["s_depth_decoder"].load_state_dict(s_depth_decoder_dict, strict=True)
            self.m_models["s_depth_decoder"].eval()
            self.m_models["s_depth_decoder"].cuda()

            # pose enc
            pose_enc_dict = torch.load(os.path.join(self.opt.teacher_weights_folder, "pose_encoder.pth"))
            self.m_models["pose_enc"] = m_networks.ResnetEncoder(18, False, num_input_images=2)
            try:
                self.m_models["pose_enc"].load_state_dict(pose_enc_dict, strict=True)
            except:
                model_dict = self.m_models["pose_enc"].state_dict()
                self.m_models["pose_enc"].load_state_dict({k: v for k, v in pose_enc_dict.items() if k in model_dict})
            self.m_models["pose_enc"].eval()
            self.m_models["pose_enc"].cuda()

            # pose dec
            pose_dec_dict = torch.load(os.path.join(self.opt.teacher_weights_folder, "pose.pth"))
            self.m_models["pose_dec"] = m_networks.PoseDecoder(self.m_models["pose_enc"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
            self.m_models["pose_dec"].load_state_dict(pose_dec_dict, strict=True)
            self.m_models["pose_dec"].eval()
            self.m_models["pose_dec"].cuda()

            # MNet encoder
            m_encoder_dict = torch.load(os.path.join(self.opt.teacher_weights_folder, "m_encoder.pth"))
            self.m_models["m_encoder"] = m_networks.FPN4(base_channels=8, scale=self.opt.prior_scale)
            self.m_models["m_encoder"].load_state_dict(m_encoder_dict, strict=True)
            self.m_models["m_encoder"].eval()
            self.m_models["m_encoder"].cuda()
    
            # reg 3d
            self.m_models["reg3d"] = m_networks.reg3d(in_channels=self.opt.reg3d_c, base_channels=self.opt.reg3d_c, down_size=3)
            reg3d_dict = torch.load(os.path.join(self.opt.teacher_weights_folder, "reg3d.pth"))
            if self.opt.adaptive_range:
                self.depth_bin_facs = reg3d_dict.get('depth_bin_facs').cuda()
                model_dict = self.m_models["reg3d"].state_dict()
                self.m_models["reg3d"].load_state_dict({k: v for k, v in reg3d_dict.items() if k in model_dict})
            else:
                self.m_models["reg3d"].load_state_dict(reg3d_dict, strict=True)
            self.m_models["reg3d"].eval()
            self.m_models["reg3d"].cuda()
            
            #convex upsample
            self.m_models["up"] = m_networks.convex_upsample_layer(feature_dim=8*2**self.opt.prior_scale, scale=self.opt.prior_scale)
            up_dict = torch.load(os.path.join(self.opt.teacher_weights_folder, "up.pth"))
            self.m_models["up"].load_state_dict(up_dict, strict=True)
            self.m_models["up"].eval()
            self.m_models["up"].cuda()
            
            self.m_backprojector = BackprojectDepth(batch_size=self.opt.num_depth_bins, height=self.opt.height//2**(self.opt.prior_scale), width=self.opt.width//2**(self.opt.prior_scale))
            self.m_projector = Project3D(batch_size=self.opt.num_depth_bins, height=self.opt.height//2**(self.opt.prior_scale), width=self.opt.width//2**(self.opt.prior_scale))
            self.m_backprojector.to(self.device)
            self.m_projector.to(self.device)
            self.matching_ids = [0, -1]
            
            self.mcs_criterion = None
            if self.opt.msc_loss_type=="l1":
                self.mcs_criterion = lambda x, y, mask: torch.abs(x[mask] - y[mask]).mean()
            elif self.opt.msc_loss_type=="ssim":
                self.mcs_criterion = lambda x, y, mask: 0.85*(self.ssim(x, y))[mask].mean() + 0.15*torch.abs(x - y).mean()
            elif self.opt.msc_loss_type=="silog":
                self.variance_focus=self.opt.variance_focus
                self.mcs_criterion = self.silog_loss



        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
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
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0 and self.epoch>=self.opt.start_save_epochs:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        #self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                #self.val()

            self.step += 1
        
        self.model_lr_scheduler.step()
    
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)          
        
        
        features = self.models["s_encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["s_depth"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        if self.opt.use_many_frame_depth:
            with torch.no_grad():
                single_feature = self.m_models["s_encoder"](inputs[("color_aug", 0, 0)])
                m_outputs = self.m_models["s_depth_decoder"](single_feature)
                disp_prior = F.interpolate(m_outputs[("disp", 0)], [self.opt.height//(2**self.opt.prior_scale), self.opt.width//(2**self.opt.prior_scale)], mode="nearest")
                disp_scaled = 1/self.opt.max_depth + disp_prior * (1/self.opt.min_depth - 1/self.opt.max_depth)
                depth_prior = 1 / disp_scaled
                
                
                pose_inputs = [self.m_models["pose_enc"](torch.cat([inputs[("color_aug", -1, 0)], inputs[("color_aug", 0, 0)]], 1))]
                axisangle, translation = self.m_models["pose_dec"](pose_inputs)

                m_outputs[("cam_T_cam", 0, -1)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
                relative_poses = torch.stack([m_outputs[("cam_T_cam", 0, -1)]], 1)
                
                
                # MNet feature extraction
                ref_match_feat, ref_context_feat = self.m_models["m_encoder"](inputs["color_aug", 0, 0])
                src_match_feats = []
                for f_i in self.matching_ids[1:]:
                #for f_i in self.opt.frame_ids[1:]:
                    src_match_feat, _ = self.m_models["m_encoder"](inputs["color_aug", f_i, 0])
                    src_match_feats.append(src_match_feat)
                
                
                if self.opt.adaptive_range:
                    s_depth_prior = schedule_depth_rangev3(depth_prior,
                                        ndepth=self.opt.num_depth_bins, 
                                        scale_facs=self.depth_bin_facs)    # (B,D,H,W)
                else:
                    s_depth_prior = schedule_depth_rangev2(depth_prior,
                                                    ndepth=self.opt.num_depth_bins, 
                                                    scale_fac=self.opt.depth_bin_fac)
                
                
                # MNet cost volume
                cor_feats = generate_costvol_v1(ref_match_feat, src_match_feats[0],
                                    inputs[('K', 2)], inputs[('inv_K', 2)],
                                    s_depth_prior, relative_poses[:, 0:1],
                                    self.opt.num_depth_bins,
                                    self.opt.reg3d_c,
                                    self.m_backprojector,
                                    self.m_projector)
        
                cost_prob = self.m_models["reg3d"](cor_feats)  # B D H W
                cost_prob = F.softmax(cost_prob, 1)
                
                depth_m = localmax_v1(cost_prob, 1, self.opt.num_depth_bins, s_depth_prior)
                depth_m = self.m_models["up"](depth_m, ref_context_feat)
                outputs['depth_m'] = depth_m.unsqueeze(1)
                
                if self.opt.msc_min_reproj_mask:
                    depth = outputs['depth_m']
                    reprojection_losses = []
                    
                    pose_inputs = [self.m_models["pose_enc"](torch.cat([inputs[("color_aug", 0, 0)], inputs[("color_aug", 1, 0)]], 1))]
                    axisangle, translation = self.m_models["pose_dec"](pose_inputs)
                    m_outputs[("axisangle", 0, 1)] = axisangle
                    m_outputs[("translation", 0, 1)] = translation
                    m_outputs[("cam_T_cam", 0, 1)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False)
                        
                    for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                        T = m_outputs[("cam_T_cam", 0, frame_id)]
                        cam_points = self.backproject_depth[0](
                            depth, inputs[("inv_K", 0)])
                        pix_coords = self.project_3d[0](
                            cam_points, inputs[("K", 0)], T)

                        pred = F.grid_sample(
                                inputs[("color", frame_id, 0)],
                                pix_coords,
                                padding_mode="border",
                                align_corners = True)
                        reprojection_losses.append(self.compute_reprojection_loss(pred, inputs[("color", 0, 0)]))

                    reprojection_losses = torch.cat(reprojection_losses, 1)
                    outputs[("reproj_loss_m")], idxs = torch.min(reprojection_losses, dim=1)
        
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            
            pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)    
            pose_inputs = [self.models["pose_encoder"](pose_inputs)]           

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

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

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            
            disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("s_depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)
                
                    

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

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

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                identity_reprojection_loss = identity_reprojection_losses


            reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            if self.opt.use_many_frame_depth and self.epoch >= self.opt.start_msc_epochs:
                if self.opt.disable_reprojection_loss:
                    loss = 0
                sf_depth = outputs[("s_depth", 0, scale)]
                mf_depth = outputs['depth_m']
                if self.opt.msc_min_reproj_mask:
                    mask = outputs[("reproj_loss_m")] < to_optimise
                    mask = mask.unsqueeze(1)
                else:
                    mask = torch.ones_like(sf_depth)
                    mask = mask==1
                
                loss += self.opt.msc_loss_weight * self.mcs_criterion(sf_depth, mf_depth, mask)
                       
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
 
    def silog_loss(self, depth_est, depth_gt, mask):
        """Compute scale invariant loss
        """
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2))
          
    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("s_depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

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
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
