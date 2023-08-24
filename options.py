import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MISMOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="options for MISM")

        # Dataset setting
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 )
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="eigen_zhou")
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--matching_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1])

        
        
        
        # Network setting
        self.parser.add_argument("--snet",
                                 type=str,
                                 help="models to load: resnet (resnet-based monodepth2) or hrnet (hrnet-based radepth)",
                                 default="hrnet",
                                 choices=["resnet", "hrnet"])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18])
        self.parser.add_argument('--convex_up',
                                 default="True",
                                 action='store_true')
        self.parser.add_argument('--load_pose',
                                 action='store_true')
        
        
        # Opt options
        self.parser.add_argument("--ssim_lw",
                                 type=float,
                                 default=0.85)
        self.parser.add_argument("--mask_lw",
                                 type=float,
                                 default=10)        
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--multi_frame_disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--pytorch_random_seed",
                                 default=None,
                                 type=int)
        self.parser.add_argument("--lr_fac",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument('--num_matching_frames',
                                 help='Sets how many previous frames to load to build the cost'
                                      'volume',
                                 type=int,
                                 default=1)              
        self.parser.add_argument("--reg3d_c",
                                 type=int,
                                 default=16)            
        self.parser.add_argument("--log",
                                 action="store_true")  
                         
        self.parser.add_argument("--prior_scale",
                                 type=int,
                                 default=2)
        self.parser.add_argument("--norm_radius",
                                 type=int,
                                 default=1)
        self.parser.add_argument("--schedule_type",
                                 type=str,
                                 default='inverse')
        
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of the folder to load")
        self.parser.add_argument("--snet_weights_folder",
                                 type=str)
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["m_encoder", "reg3d", "up", "s_depth", "s_encoder", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="./log")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="teacher_hrnet")
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--save_intermediate_models",
                                 help="if set, save the model each time we log to tensorboard",
                                 action='store_true')
        self.parser.add_argument("--start_save_epochs",
                                 type=int,
                                 default=-1)
        
        # EVALUATION options
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original snetdepth paper",
                                 action="store_true") 
    
        # DDP
        self.parser.add_argument("--local_rank", default=0,type=int)
        self.parser.add_argument("--ddp", 
                                  help="Use DistributedDataParallel",
                                  action="store_true")
        
        
        # PADS setting
        self.parser.add_argument("--adaptive_range", 
                                  help="update adaptive depth range",
                                  default=True,
                                  action='store_true')
        self.parser.add_argument("--delta_fac",
                                 type=float,
                                 default=1.2)
        self.parser.add_argument("--num_depth_bins",
                                 type=int,
                                 default=16)
        self.parser.add_argument("--depth_bin_fac",
                                 type=float,
                                 default=1.0)
        
        
        #Disitll with the consistency between multi-frame depth and single-frame depth
        self.parser.add_argument("--use_many_frame_depth",
                                 help="if set, uses many frame depth to guide singel-frame depth estimation",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--msc_loss_type",
                                 type=str,
                                 default="silog",
                                 choices=["l1", "ssim", "silog"],
                                 help="which multi-frame and single-frame depth consistency loss type")
        self.parser.add_argument("--msc_min_reproj_mask",
                                 help="msc_loss works when the multi-frame depth could improve the photometric error",
                                 default=True,
                                 action="store_true")
        self.parser.add_argument("--start_msc_epochs",
                                 type=int,
                                 default=-1)
        self.parser.add_argument("--msc_loss_weight",
                                 help="if set multiplies msc loss by this number",
                                 type=float,
                                 default=1e-1)
        self.parser.add_argument("--variance_focus",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--disable_reprojection_loss",
                                 help="if set will only compute msc loss",
                                 action="store_true")
        self.parser.add_argument("--teacher_weights_folder",
                                 help="if set will load the teacher weights from this folder",
                                 type=str)
        
        
        
        
             

        
     
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
