python train.py --experiment_name cris_openpose_AEC_lr_0.0005 --training_set_path ../DATASETS/AEC--openpose-Train.hdf5 --validation_set_path ../DATASETS/AEC--openpose-Val.hdf5  --testing_set_path ../DATASETS/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 20 --keypoints_model openpose --lr 0.0005

python train.py --experiment_name cris_openpose_AEC_lr_0.001 --training_set_path ../DATASETS/AEC--openpose-Train.hdf5 --validation_set_path ../DATASETS/AEC--openpose-Val.hdf5  --testing_set_path ../DATASETS/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 20 --keypoints_model openpose --lr 0.001


python train.py --experiment_name cris_openpose_AEC_lr_0.002 --training_set_path ../DATASETS/AEC--openpose-Train.hdf5 --validation_set_path ../DATASETS/AEC--openpose-Val.hdf5  --testing_set_path ../DATASETS/AEC--openpose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 20 --keypoints_model openpose --lr 0.002


'''
python train.py --experiment_name cris_mediapipe_AEC_TEST_model --training_set_path ../DATASETS/AEC--mediapipe-Train.hdf5 --validation_set_path ../DATASETS/AEC--mediapipe-Val.hdf5  --testing_set_path ../DATASETS/AEC--mediapipe-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 5 --keypoints_model mediapipe


python train.py --experiment_name cris_wholepose_AEC_TEST_model2 --training_set_path ../DATASETS/AEC--wholepose-Train.hdf5 --validation_set_path ../DATASETS/AEC--wholepose-Val.hdf5  --testing_set_path ../DATASETS/AEC--wholepose-Val.hdf5 --hidden_dim 142 --num_classes 28 --epochs 2 --keypoints_model wholepose
'''

