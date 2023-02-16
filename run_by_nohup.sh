nohup python train.py --wandb --epoch 3000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 3 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model mediapipe > 24-11-2022-3_datasets-29-epoch_3000_mediapipe_lr_0.0005_coders_3_ffdim_8.txt &
nohup python train.py --wandb --epoch 3000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 3 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model mediapipe > 24-11-2022-3_datasets-29-epoch_5000_mediapipe_lr_0.00005_coders_3_ffdim_8.txt &
nohup python train.py --wandb --epoch 3000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 3 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model mediapipe > 24-11-2022-3_datasets-29-epoch_2000_mediapipe_lr_0.00005_coders_3_ffdim_8.txt &


MULTI_GPU=1 CUDA_VISIBLE_DEVICES=0 nohup python train.py --wandb --epoch 15000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset AEC-DGI156-DGI305 --keypoints_model mediapipe > log_AEC-DGI156-DGI305_29_00005_mediapipe.txt &
