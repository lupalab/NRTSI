# NRTSI
This is the official repository of the paper [NRTSI: Non-Recurrent Time Series Imputation](https://arxiv.org/abs/2102.03340)

```ruby
@inproceedings{shan2023nrtsi,
  title={Nrtsi: Non-recurrent time series imputation},
  author={Shan, Siyuan and Li, Yang and Oliva, Junier B},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

NRTSI is a state-of-the-art time series imputation model that this broadly applicable to regularly-sampled time series, irregularly-sampled time series, time series with partially observed dimensions and time series requiring stochastic imputation.

All the results of NRTSI in the paper can be reproduced via the codes in this repository. We organize the codes into 4 folders for handling regularly-sampled time series, irregularly-sampled time series, time series with partially observed dimensions and time series requiring stochastic imputation. Inside each folder, their is a detailed readme file to instruct how to train and impute. Before you get started, make sure you have installed the dependencies in `requirements.txt`

The figure below shows the power of NRTSI on imputing billiard trajectories with only 5 time points observed.
![image](https://github.com/lx4ri6y78w/NRTSI/blob/main/illustration1.png?raw=true)

