# AEC-29
#nohup python train.py --wandb --epoch 7000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset AEC --keypoints_model mediapipe > log_AEC_29_00005_mediapipe.txt &
#nohup python train.py --wandb --epoch 7000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset AEC --keypoints_model wholepose > log_AEC_29_00005_wholepose.txt &
#nohup python train.py --wandb --epoch 7000 --learning_rate 0.00005 --keypoints_number 29 --num_coder_layers 1 --dim_feedforward 32 --device 0 --seed 1 --dataset AEC --keypoints_model openpose > log_AEC_29_00005_openpose.txt &

# AEC-71
nohup python train.py --wandb --epoch 10000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model mediapipe > log_AEC_71_00005-mediapipe.txt & 
nohup python train.py --wandb --epoch 10000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model wholepose > log_AEC_71_00005-wholepose.txt &
nohup python train.py --wandb --epoch 10000 --learning_rate 0.00005 --keypoints_number 71 --num_coder_layers 1 --dim_feedforward 8 --device 0 --seed 1 --dataset AEC --keypoints_model openpose > log_AEC_71_00005-openpose.txt &
