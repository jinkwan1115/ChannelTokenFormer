# [ICLR 2026] ChannelTokenFormer
This repository is the official implementation of Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model in the paper, run this command:

```train
bash ./scripts_practical/ChannelTokenFormer/CTF_ETT1_practical.sh
```
These scripts include evaluation also.

## Evaluation

To evaluate model on SolarWind-missing scenarios, you should first train and save a checkpoint with the practical setting, then run missing-scenario inference.

1. Train with SolarWind practical setting and generate checkpoint files:

```train
bash ./scripts_practical/ChannelTokenFormer/CTF_SW_practical.sh
```

2. Confirm the saved checkpoint directory from the training output (or your configured checkpoints path).

3. Set that checkpoint directory in the missing evaluation script/config, then run:

```eval
bash ./scripts_practical/ChannelTokenFormer/CTF_SW_missing.sh
```
`CTF_SW_missing.sh` requires a valid pretrained checkpoint path from Step 1.

## Contributing

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- CrossGNN (https://github.com/hqh0728/CrossGNN)
