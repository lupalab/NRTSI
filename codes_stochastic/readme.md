# This repository contains the codes to train and impute NRTSI on the Football Player Trajectory dataset for stochastic imputation. 
This dataset is collected from 2021 Big Data Bowl data (https://www.kaggle.com/c/nfl-big-data-bowl-2021).
This dataset contains 9,543 time series with 50 regularlysampled time points each. The sampling rate is 10 Hz. Each
time series contains the 2D trajectories of six offensive players and is therefore 12-dimensional. During training and
testing, we treat all players in a time series equally and randomly permute their orders. 8,000 time series are randomly
selected for training and the others are used for testing.

If you want to train on this dataset, please use the following link to download the data files and save them to "./data". 

nfl training set: https://www.dropbox.com/s/5fn3dvsmbu8wheb/nfl_train.npy?dl=0

nfl testing set: https://www.dropbox.com/s/u7y9gx9yi34lg8n/nfl_test.npy?dl=0


Training Command:
```
python train.py --cfg_file=./params.json
```

After Training, you can use the following command to impute 

Imputation Command:
```
python impute.py --cfg_file=./params.json
```
If you want to visualize the imputed trajectories, the imputation command is:
```
python impute.py --cfg_file=./params.json --save_fig=1
```
The saved figures can be found at ./log/xxx/impute
