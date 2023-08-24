import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from utils import readlines
from options import MISMOptions
import datasets
from MNet import networks
from layers import transformation_from_parameters, disp_to_depth, schedule_depth_rangev2, BackprojectDepth, Project3D, \
    generate_costvol_v1, localmax_v1, schedule_depth_rangev3

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
        
    print("Using ckpt: {}".format(opt.load_weights_folder))
    backprojector = BackprojectDepth(batch_size=opt.num_depth_bins,
                                        height=opt.height//(2**opt.prior_scale),
                                        width=opt.width//(2**opt.prior_scale))
    projector = Project3D(batch_size=opt.num_depth_bins,
                                    height=opt.height//(2**opt.prior_scale),
                                    width=opt.width//(2**opt.prior_scale))
    backprojector.cuda()
    projector.cuda()
    
    frames_to_load = opt.matching_ids
    HEIGHT, WIDTH = opt.height, opt.width

    img_ext = '.png' if opt.png else '.jpg'
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))
    filenames = readlines("splits/"+ opt.eval_split + "/test_files.txt")


    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                        HEIGHT, WIDTH,
                                        frames_to_load, 4,
                                        is_train=False,
                                        img_ext=img_ext)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)



    # setup models
    # single-frame encoder
    try:
        s_encoder_path = os.path.join(opt.load_weights_folder, "s_encoder.pth")
    except:
        s_encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    s_encoder_dict = torch.load(s_encoder_path)
    if opt.snet=="resnet":
        s_encoder = networks.ResnetEncoder(num_layers=opt.num_layers, pretrained=False)
    elif opt.snet=="hrnet":
        s_encoder = networks.hrnet18(False)
    try:
        s_encoder.load_state_dict(s_encoder_dict, strict=True)
    except:
        model_dict = s_encoder.state_dict()
        s_encoder.load_state_dict({k: v for k, v in s_encoder_dict.items() if k in model_dict})
    s_encoder.eval()
    s_encoder.cuda()

    # single-frame decoder
    try:
        s_decoder_path = os.path.join(opt.load_weights_folder, "s_depth.pth")
    except:
        s_decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    s_depth_decoder_dict = torch.load(s_decoder_path)
    if opt.snet=="resnet":
        s_depth_decoder = networks.DepthDecoder(s_encoder.num_ch_enc)
    elif opt.snet=="hrnet":
        s_depth_decoder = networks.DepthDecoder_MSF(s_encoder.num_ch_enc, [0], num_output_channels=1)
    s_depth_decoder.load_state_dict(s_depth_decoder_dict, strict=True)
    s_depth_decoder.eval()
    s_depth_decoder.cuda()

    # pose enc
    pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
    pose_enc = networks.ResnetEncoder(opt.num_layers, False, num_input_images=2)
    try:
        pose_enc.load_state_dict(pose_enc_dict, strict=True)
    except:
        model_dict = pose_enc.state_dict()
        pose_enc.load_state_dict({k: v for k, v in pose_enc_dict.items() if k in model_dict})
    pose_enc.eval()
    pose_enc.cuda()

    # pose dec
    pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)
    pose_dec.eval()
    pose_dec.cuda()

    # multi-frame depth encoder
    m_encoder_dict = torch.load(os.path.join(opt.load_weights_folder, "m_encoder.pth"))
    m_encoder = networks.FPN4(base_channels=8, scale=opt.prior_scale)
    m_encoder.load_state_dict(m_encoder_dict, strict=True)
    m_encoder.eval()
    m_encoder.cuda()

    # reg 3d
    if opt.num_depth_bins >=8:
        reg3d = networks.reg3d(in_channels=opt.reg3d_c, base_channels=opt.reg3d_c, down_size=3)  # FIXME hardcoded
    else:
        reg3d = networks.reg2d(input_channel=opt.reg3d_c, base_channel=opt.reg3d_c)  # FIXME hardcoded
    reg3d_dict = torch.load(os.path.join(opt.load_weights_folder, "reg3d.pth"))
    if opt.adaptive_range:
        depth_bin_facs = reg3d_dict.get('depth_bin_facs').cuda()
        model_dict = reg3d.state_dict()
        reg3d.load_state_dict({k: v for k, v in reg3d_dict.items() if k in model_dict})
    else:
        reg3d.load_state_dict(reg3d_dict, strict=True)
    reg3d.eval()
    reg3d.cuda()
    
    # convex upsample
    if opt.convex_up:
        uplayer = networks.convex_upsample_layer(feature_dim=8*2**opt.prior_scale, scale=opt.prior_scale)
        up_dict = torch.load(os.path.join(opt.load_weights_folder, "up.pth"))
        uplayer.load_state_dict(up_dict, strict=True)
        uplayer.eval()
        uplayer.cuda()



    pred_disps_z = []
    pred_disps_single = []
    
    
    print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
    # do inference
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            input_color = data[('color', 0, 0)].cuda()
            output = s_encoder(input_color)
            output = s_depth_decoder(output)
                
            # predict poses
            pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
            if torch.cuda.is_available():
                pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
            for fi in frames_to_load[1:]:
                if fi < 0:
                    pose_inputs = [pose_feats[fi], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[fi]]

                pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                axisangle, translation = pose_dec(pose_inputs)
                pose = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=fi<0)
                data[('relative_pose', fi)] = pose
            relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
            relative_poses = torch.stack(relative_poses, 1)

            if torch.cuda.is_available():
                relative_poses = relative_poses.cuda()

            ref_match_feas, ref_context_feat = m_encoder(input_color)
            src_match_feats = []
            for f_i in opt.matching_ids[1:]:
                src_match_feat, _ = m_encoder(data["color_aug", f_i, 0].cuda())
                src_match_feats.append(src_match_feat)



            disp_prior = F.interpolate(output[("disp", 0)], [opt.height//(2**opt.prior_scale),opt.width//(2**opt.prior_scale)], mode="nearest").detach()
            disp_scaled = 1/opt.max_depth + disp_prior * (1/opt.min_depth - 1/opt.max_depth)
            depth_prior = 1 / disp_scaled

            
            
            
            if opt.adaptive_range:
                mono_depth_prior_z = schedule_depth_rangev3(depth_prior,
                                                    ndepth=opt.num_depth_bins, 
                                                    scale_facs=depth_bin_facs,
                                                    type=opt.schedule_type)
            else:
                mono_depth_prior_z = schedule_depth_rangev2(depth_prior,
                                                    ndepth=opt.num_depth_bins, 
                                                    scale_fac=opt.depth_bin_fac,
                                                    type=opt.schedule_type)
            
            cor_feats_z = generate_costvol_v1(ref_match_feas, src_match_feats[0],
                                    data[('K', 2)].cuda(), data[('inv_K', 2)].cuda(),
                                    mono_depth_prior_z, relative_poses[:, 0:1],
                                    opt.num_depth_bins,
                                    opt.reg3d_c,
                                    backprojector,
                                    projector)
            
            cost_prob_z = reg3d(cor_feats_z)  # B D H W
            cost_prob_z = F.softmax(cost_prob_z, 1)

            #depth_m_z = localmax_v1(cost_prob_z, opt.norm_radius, opt.num_depth_bins, torch.flip(mono_depth_prior_z, [1]))
            depth_m_z = localmax_v1(cost_prob_z, opt.norm_radius, opt.num_depth_bins, mono_depth_prior_z)
            

            if opt.convex_up:
                depth_m_z = uplayer(depth_m_z, ref_context_feat)

            pred_disps_z.append(1/depth_m_z.cpu().numpy())
            pred_disp_single, pred_single_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp_single = pred_disp_single.cpu()[:, 0].numpy()
            pred_disps_single.append(pred_disp_single)

        
        pred_disps_z = np.concatenate(pred_disps_z)
        pred_disps_single = np.concatenate(pred_disps_single)


    gt_path = "splits/" + opt.eval_split + "/gt_depths.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    errors_z = []
    errors_single = []

    
    ratios1 = []
    ratios2 = []
    ratios3 = []

    for i in tqdm(range(pred_disps_single.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp_z = np.squeeze(pred_disps_z[i])
        pred_disp_single = np.squeeze(pred_disps_single[i])
        
        pred_disp_single = cv2.resize(pred_disp_single, (gt_width, gt_height))
        pred_disp_z = cv2.resize(pred_disp_z, (gt_width, gt_height))
        
        pred_depth_z = 1 / pred_disp_z
        pred_depth_single = 1 / pred_disp_single
        

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth_z = pred_depth_z[mask]
        pred_depth_single = pred_depth_single[mask]
        gt_depth = gt_depth[mask]


        if not opt.disable_median_scaling:
            ratio1 = np.median(gt_depth) / np.median(pred_depth_single)
            ratios1.append(ratio1)
            pred_depth_single *= ratio1
            ratio2 = np.median(gt_depth) / np.median(pred_depth_z)
            ratios2.append(ratio2)
            pred_depth_z *= ratio2
 


        pred_depth_z[pred_depth_z < MIN_DEPTH] = MIN_DEPTH
        pred_depth_z[pred_depth_z > MAX_DEPTH] = MAX_DEPTH
        pred_depth_single[pred_depth_single < MIN_DEPTH] = MIN_DEPTH
        pred_depth_single[pred_depth_single > MAX_DEPTH] = MAX_DEPTH
        

        this_m_err_z = compute_errors(gt_depth, pred_depth_z)
        this_single_err = compute_errors(gt_depth, pred_depth_single)
        

        errors_z.append(this_m_err_z)
        errors_single.append(this_single_err)
        

    mean_errors_z = np.array(errors_z).mean(0)
    mean_errors_single = np.array(errors_single).mean(0)

    #print('depth_bin_fac:', opt.depth_bin_fac, 'use_dcg',opt.dcg,)
    print('sinlge-frame results:')
    if not opt.disable_median_scaling:
        ratios = np.array(ratios1)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors_single.tolist()) + "\\\\")
    print("\n")


    print('multi-frame results:')
    if not opt.disable_median_scaling:
        ratios = np.array(ratios2)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors_z.tolist()) + "\\\\")
    print("\n")

if __name__ == "__main__":
    options = MISMOptions()
    evaluate(options.parse())