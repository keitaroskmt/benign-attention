# Benign Overfitting in Token Selection of Attention Mechanism

This repository contains the official implementation of our ICML 2025 paper: [Benign Overfitting in Token Selection of Attention Mechanism](https://arxiv.org/abs/2409.17625)

<p align="center">
      <img src="https://github.com/keitaroskmt/benign-attention/blob/94f5c60a20acb0a46b6cd0812842ee155eb63af8/img/figure1.png", width=90%, height=90%>
<!p>

## Requirements

The code used Python 3.13 and PyTorch 2.6.
We recommend using [`uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for package management.

To set up the environment, run:

```bash
uv init
uv sync
```

If you are new to [Weights & Biases](https://wandb.ai/site), please login:

```bash
wandb login
```

## Reproducing Results

### Synthetic Experiments

#### Our Model Setup (Figure 3 and 5)

To run the synthetic experiments in our paper, run:

```python
uv run main.py wandb.entity=your_account
```

**Note:** Replace `your_account` with your actual W&B account name.

To run with different hyperparameter, you can either specify them in the command line or modify `config/main.yaml`.

#### One-layer Transformer Encoder (Figure 6 and 7)

To run the synthetic experiments with one-layer transformer encoder in the appendix, run:

```python
uv run main_one_layer.py wandb.entity=your_account
```

To run with different hyperparameter, you can either specify them in the command line or modify `config/main_one_layer.yaml`.

### Real-world Experiments (Table 2 and 6, Figure 8 and 9)

We support the real-world experiments on both image dataset (MNIST, CIFAR10, STL10, PneumoniaMNIST, and BreastMNIST) and text dataset (AG-news, SST-2, and TREC).

To run experiments on an image dataset (e.g., MNIST):

```python
uv run main_vit_finetune.py dataset=mnist wandb.entity=your_account
```

To run with different hyperparameter, you can either specify them in the command line or modify `config/main_vit.yaml`.
The dataset argument supports `mnist`, `cifar10`, `stl10`, `pneumoniamnist`, and `breastmnist`.

To run experiments on a text dataset (e.g., AG-news):

```python
uv run main_bert_finetune.py dataset=agnews wandb.entity=your_account
```

To run with different hyperparameter, you can either specify them in the command line or modify `config/main_bert.yaml`.
The dataset argument supports `agnews`, `sst2`, and `trec`.


### Citation

If you find our work useful for your research, please cite using this BibTeX:

```BibTeX
@article{sakamoto2025benign,
      title={Benign overfitting in token selection of attention mechanism},
      author={Sakamoto, Keitaro and Sato, Issei},
      booktitle={International Conference on Machine Learning},
      year={2025}
}
```
