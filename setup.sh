sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

pip install -r requirements.txt

#python train.py --experiment_name cris_no_normalization --training_set_path datasets/WLASL100_train_25fps.csv --validation_set_path datasets/WLASL100_val_25fps.csv --testing_set_path datasets/WLASL100_test_25fps.csv


# python train.py --experiment_name cris_openpose_AEC_TEST --training_set_path ../SignLanguage/Data/H5/AEC--openpose-train.csv --validation_set_path ../SignLanguage/Data/H5/AEC--openpose-val.csv  --testing_set_path ../SignLanguage/Data/H5/AEC--openpose-val.csv --hidden_dim 142 --num_classes 28

# python train.py --experiment_name cris_openpose_AEC_TEST_200 --training_set_path ../SignLanguage/Data/H5/AEC--openpose-train.csv --validation_set_path ../SignLanguage/Data/H5/AEC--openpose-val.csv  --testing_set_path ../SignLanguage/Data/H5/AEC--openpose-val.csv --hidden_dim 142 --num_classes 28 --epochs 200
