# This repository contains the codes to train and impute NRTSI on the two partially observed datasets (i.e. air quality and gas)

please use the following link to download the npy files and save them to "./data". 

These two datasets is used in RDIS (https://arxiv.org/abs/2010.10075). We strictly follow their setup and use the same random seed the split the whole dataset into a training set, a testing set and a validation set to make the results comparable.

Air quality training set: https://www.dropbox.com/s/v29re767xew0070/air_quality_train.npy?dl=0

Air quality validation set: https://www.dropbox.com/s/xvrn7fxn974ckvt/air_quality_val.npy?dl=0

Air quality testing set: https://www.dropbox.com/s/ac0ix4kto2iyqxl/air_quality_test.npy?dl=0

Gas training set: https://www.dropbox.com/s/59a7yoirec0d2k9/gas_train.npy?dl=0

Gas validation set: https://www.dropbox.com/s/lnzuhp4xo6as3d0/gas_val.npy?dl=0

Gas testing set: https://www.dropbox.com/s/omp2ujbtr48iaka/gas_test.npy?dl=0

Training Command:
```
python train.py --cfg_file=./params.json
```
To choose which dataset to run, please change the "dataset" parameter inside "params.json". You can set "dataset" to "gas" or "air_quality".

After Training, you can use the following command to impute 

Imputation Command:
```
python impute.py --cfg_file=./params.json
```

