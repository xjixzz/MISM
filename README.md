# MISM
  

> **Exploring the Mutual Influence between Self-Supervised Single-Frame and Multi-Frame Depth Estimation**

> [arxiv pdf ](https://arxiv.org/abs/2304.12685)
#### 1. Install
```
conda create -n MISM python=3.7 -y
conda activate MISM
conda install pytorch==1.10.1 torchvision==0.2.1 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda install opencv=3.4.2 pillow=8.4.0 -y
pip install -i Pillow==8.4.0 matplotlib==3.1.2 scikit-image==0.16.2 tqdm==4.57.0 tensorboardX==1.5 protobuf==3.19.1 timm==0.4.12 yacs==0.1.8 ptflops==0.6.9 thop
```

#### 2. Prepare dataset
Follow [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) to prepare the KITTI dataset

#### 3. Train on the KIITTI dataset
download pretrained [HRNet18](https://github.com/HRNet/HRNet-Image-Classification/releases/download/PretrainedWeights/HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar)
```
git clone https://github.com/xjixzz/MISM.git
cd mism
mkdir pretrained_models
wget -P ./pretrained_models/ https://github.com/HRNet/HRNet-Image-Classification/releases/download/PretrainedWeights/HRNet_W18_C_cosinelr_cutmix_300epoch.pth.tar
```

(1) train the teacher model on the KITTI dataset
```
CUDA_VISIBLE_DEVICES=${device} python -m torch.distributed.launch --nproc_per_node 1 --master_port 11000 -m MNet.train_teacher \
    --data_path $data_path \
    --scales 0 \
    --log_dir "${log_dir}"  \
    --model_name ${teacher_model} \
    --png \
    --learning_rate 2e-4
```

(2) distillation learning
```
teacher_weights_folder="${log_dir}/${teacher_model}/models/last"
CUDA_VISIBLE_DEVICES=${device} python -m SNet.train_student \
    --scales 0 \
    --data_path ${data_path} \
	--log_dir ${log_dir} \
    --png \
	--teacher_weights_folder ${teacher_weights_folder} \
	--model_name ${student_model}
```

#### 4. Test on kitti
(1) test on the eigen split of kitti with the raw GT depth
export gt depth of the eigen split
```
python export_gt_depth.py --data_path ${data_path} --split eigen
```
```
distilled_weights_folder="${log_dir}/${student_model}/models/weights_19"
cp "$teacher_weights_folder/m_encoder.pth" $distilled_weights_folder
cp "$teacher_weights_folder/reg3d.pth" $distilled_weights_folder
cp "$teacher_weights_folder/up.pth" $distilled_weights_folder
CUDA_VISIBLE_DEVICES=${device} python -m MNet.evaluate_teacher \
    --data_path $data_path \
    --load_weights_folder ${distilled_weights_folder} \
    --scales 0 \
    --png \
    --batch_size 1
```
The test results of our [model](https://drive.google.com/file/d/1NYYimMgYXA6eVa0Zxc0kbUGOGhLOAQ6j/view?usp=sharing) are as follows:
| abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
|---------|--------|-------|----------|-------|-------|-------|
| 0.086   | 0.613  | 4.096 | 0.165    | 0.915 | 0.969 | 0.985 |

(2) test with the improved GT depth
export gt depth of the eigen_benchmark split
```
python export_gt_depth.py --data_path ${data_path} --split eigen_benchmark
```
```
CUDA_VISIBLE_DEVICES=${device} python -m MNet.evaluate_teacher \
    --data_path $data_path \
    --load_weights_folder ${distilled_weights_folder} \
    --scales 0 \
    --png \
    --batch_size 1 \
    --eval_split eigen_benchmark 
```
The test results of our our [model](https://drive.google.com/file/d/1NYYimMgYXA6eVa0Zxc0kbUGOGhLOAQ6j/view?usp=sharing) on the eigen_benchmark are as follows:
| abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
|---------|--------|-------|----------|-------|-------|-------|
| 0.058   | 0.302  | 3.070 | 0.098    | 0.955 | 0.992 | 0.998 |

#### Acknowledgment
Our implementation is mainly based on [monodepth2](https://github.com/nianticlabs/monodepth2), [RA-Depth](https://github.com/hmhemu/RA-Depth) and [MOVEDepth](https://github.com/JeffWang987/MOVEDepth). Thanks for their authors.
