# Polarization2RGB


Download Dataset
--------------------
In this work, we establish a comprehensive benchmark for RGB-to-polarization image estimation. 
To ensure fair and consistent evaluation across methods, we adopt one of the latest publicly available RGB-polarization datasets and standardize the evaluation protocol.

Dataset reference:
@inproceedings{jeon2024spectral,
  title     = {Spectral and Polarization Vision: Spectro-polarimetric Real-world Dataset},
  author    = {Jeon, Yujin and Choi, Eunsue and Kim, Youngchan and Moon, Yunseong and Omer, Khalid and Heide, Felix and Baek, Seung-Hwan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {22098--22108},
  year      = {2024}
}

Please download the dataset from the Hugging Face repository:  
https://huggingface.co/datasets/jyj7913/spectro-polarimetric

The dataset should follow the structure below:

<root_dir>/
├── 0000/
│   ├── s0.png
│   ├── s1.png
│   ├── s2.png
│   └── s3.png 
├── 0001/
│   ├── s0.png
│   ├── s1.png
│   ├── s2.png
│   └── s3.png
├── ...

Training
---------------
To train our model:

1. Select a model in `configs/polarization.yml` (e.g., MAE, Restormer).
2. Set the dataset path in **line 22** of `configs/polarization.yml`.
3. Run one of the following commands (requires 4 × 24GB NVIDIA GPUs).

Below are example commands for two models:

Example – MAE:
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u -m torch.distributed.run \
--nproc_per_node=4 --master-port=23369 train.py \
--config "polarization.yml" > log_train_mae.txt 2>&1 &

Example – Restormer:
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u -m torch.distributed.run \
--nproc_per_node=4 --master-port=23369 train.py \
--config "polarization.yml" > log_train_restormer.txt 2>&1 &

Evaluation
--------------
To evaluate the trained model:

Example – MAE:
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u eval_diffusion.py \
--config "polarization.yml" \
--resume './MAE/polarization.pth.tar' \
--test_set 'MAE' > log_test_MAE.txt 2>&1 &

Example – Restormer:
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u eval_diffusion.py \
--config "polarization.yml" \
--resume './Restormer/polarization.pth.tar' \
--test_set 'Restormer' > log_test_Restormer.txt 2>&1 &
