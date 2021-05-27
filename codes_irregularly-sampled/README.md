# This repository contains the codes to train and impute NRTSI on the two irregularly-sampled datasets (i.e. irregularly-sampled Billiard and irregularly-sampled Sinusoidal function)

To create the irregularly-sampled Billiard dataset, run the following command:
```
python make_irr_billiard.py
```

To create the irregularly-sampled Sinusoidal function dataset, run the following command:
```
python make_irr_sin.py
```

Training Command:
```
python train.py --cfg_file=./params.json
```
To choose which dataset to run, please change the "dataset" parameter inside "params.json". You can set "dataset" to "billiard_irr" or "sin_irr".

After Training, you can use the following command to impute 

Imputation Command:
```
python impute.py --cfg_file=./params.json
```

