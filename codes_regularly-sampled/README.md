# This repository contains the codes to train and impute NRTSI on the regularly-sampled time-series datasets (e.g. Billiard, Traffic, MuJoCo).

If you want to train on Billiard datasets, please use the following link to download the files and save them to "./data". 

Billiard training set: https://www.dropbox.com/s/k8k3h209lpyrw56/billiard_train.p?dl=0

Billiard testing set: https://www.dropbox.com/s/oxfwjls7ajym8h6/billiard_eval.p?dl=0

Note that the Billiard dataset is used in NAOMI (https://papers.nips.cc/paper/2019/file/50c1f44e426560f3f2cdcb3e19e39903-Paper.pdf) and it only contains a training set, a testing set and no validation set. Therefore, to make our results comparable to NAOMI, we use the same training set, the same testing set and no validation set.

If you want to train on Traffic and MuJoCo datasets, please use the following link to download the npy files and save them to "./data". 

Traffic training set: https://www.dropbox.com/s/jbvsk1689epskut/pems_train.npy?dl=0

Traffic testing set: https://www.dropbox.com/s/ev5qw0ygd4ckjtm/pems_test.npy?dl=0

MuJoCo training set: https://www.dropbox.com/s/pjccc2piis8g2fx/mujoco_train.npy?dl=0

MuJoCo testing set: https://www.dropbox.com/s/ktkswh77sueqfy8/mujoco_test.npy?dl=0

Note that the Traffic dataset is used in NAOMI (https://papers.nips.cc/paper/2019/file/50c1f44e426560f3f2cdcb3e19e39903-Paper.pdf) and it only contains a training set, a testing set and no validation set. Therefore, to make our results comparable to NAOMI, we use the same training set, the same testing set and no validation set. Also the MuJoCo dataset is used in LatentODE (https://arxiv.org/pdf/1907.03907.pdf) and LatentODE use 80% for training and 20% for testing (i.e. also no validation set). To make results comparable, we also strictly follow the setup of LatentODE and use the same random seed to split the whole dataset into a training set and a testing set.



Training Command:
```
python train.py --cfg_file=./params.json
```
To choose which dataset to run, please change the "dataset" parameter inside "params.json". You can set "dataset" to "billiard", "traffic" or "mujoco".

After Training, you can use the following command to impute 

Imputation Command:
```
python impute.py --cfg_file=./params.json
```
If you want to visualize the imputed trajectories, the imputation command is:
```
python impute.py --cfg_file=./params.json --save_fig=1
```
The saved figures can be found at ./log/per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss_att_d_128_model_d_1024/impute
