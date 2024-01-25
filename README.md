# BotDGT: Dynamicity-aware Social Network Bot Detection with Dynamic Graph Transformers
Implementation for the paper "BotDGT: Dynamicity-aware Social Network Bot Detection with Dynamic Graph Transformers".

BotDGT is a framework that leverages the dynamic nature of social networks to enhance the graph-based bot detection methods.

## Requirements
* torch-geometric==2.1.0
* torch==1.13.0

## Quick Start

### Dataset Preparation
The original datasets are available at [Twibot-20](https://github.com/BunsenFeng/TwiBot-20) and [Twibot-22](https://github.com/LuoUndergradXJTU/TwiBot-22). 

For each dataset:

1. Put the raw data (original .json .csv files) in the `\raw` folder

2. Put the processed data (original .pt files) in the `\processed` folder

3. Run `preprocess.py` to generate the graph snapshots.

### Train and Test
```python
# For Twibot-20, run the following commands
python main.py --dataset_name "Twibot-20" --batch_size 64 --hidden_dim 128 --weight_decay 1e-2 --structural_learning_rate 1e-4 --temporal_learning_rate 1e-5
# For Twibot-22, run the following commands
python main.py --dataset_name "Twibot-22" --batch_size 256 --hidden_dim 64 --weight_decay 5e-2 --structural_learning_rate 5e-4 --temporal_learning_rate 5e-5

```