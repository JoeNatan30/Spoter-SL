'''

############################
python train.py --experiment_name tunning/AEC/cris_openpose_AEC_lr_0.00025 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.00025

python train.py --experiment_name tunning/AEC/cris_openpose_AEC_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.0005

python train.py --experiment_name tunning/AEC/cris_openpose_AEC_lr_0.001 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.001

python train.py --experiment_name tunning/AEC/cris_openpose_AEC_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.002

python train.py --experiment_name tunning/AEC/cris_openpose_AEC_lr_0.003 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 50 --keypoints_model openpose --lr 0.003

#############################
python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.00025 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.00025

python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.0005

python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.001 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.001

python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.002


python train.py --experiment_name tunning/PUCP/cris_mediapipe_PUCP_lr_0.003 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 30 --keypoints_model mediapipe --lr 0.003

##############################
python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.00025 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.00025

python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.0005

python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.001 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.001

python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.002

python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.003 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 30 --keypoints_model wholepose --lr 0.003


python train.py --experiment_name tunning/WLASL/cris_wholepose_WLASL_lr_0.003_exp --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 3 --keypoints_model wholepose --lr 0.003

######################################

python train.py --experiment_name final/AEC/cris_openpose_AEC_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 200 --keypoints_model openpose --lr 0.0005

python train.py --experiment_name final/AEC/cris_mediapipe_AEC_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/AEC--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 200 --keypoints_model mediapipe --lr 0.0005

python train.py --experiment_name final/AEC/cris_wholepose_AEC_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/AEC--wholepose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/AEC--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/AEC--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 200 --keypoints_model wholepose --lr 0.0005

'''

##############################################
python train.py --experiment_name final/PUCP/cris_openpose_PUCP_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--openpose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--openpose-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 200 --keypoints_model openpose --lr 0.0005

python train.py --experiment_name final/PUCP/cris_mediapipe_PUCP_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 200 --keypoints_model mediapipe --lr 0.0005

python train.py --experiment_name final/PUCP/cris_wholepose_PUCP_lr_0.0005 --training_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--wholepose-Train.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/PUCP_PSL_DGI156--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 36 --epochs 200 --keypoints_model wholepose --lr 0.0005

##############################################

python train.py --experiment_name final/WLASL/cris_openpose_WLASL_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/WLASL--openpose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--openpose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--openpose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 200 --keypoints_model openpose --lr 0.002

python train.py --experiment_name final/WLASL/cris_mediapipe_WLASL_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/WLASL--mediapipe-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--mediapipe-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 200 --keypoints_model mediapipe --lr 0.002

python train.py --experiment_name final/WLASL/cris_wholepose_WLASL_lr_0.002 --training_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --validation_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5  --testing_set_path ../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 101 --epochs 200 --keypoints_model wholepose --lr 0.002
