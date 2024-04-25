
# Developer
- 2021 Yuki SENSUI, TUT Speech language processing Laboratory
- 2022 Takaaki Uno, TUT
- 2024 Ryota Nishimura, Tokushima University

# Overview
Repository of Wake Word detection model.

# Prepare data
Before running `train.py`, you need to generate each dataset.  
For example, the following command will generate a dataset for detecting words with 6 phonemes from `train_dev` and output it to `data/6/train_dev/x (y)`.

```bash
python data.py 6 train_dev
```

# Train model
```bash
python train.py
train_dev
```