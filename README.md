## Benign or Not-Benign Overfitting in Token Selection of Attention Mechanism
This repository is the official implementation of [Benign or Not-Benign Overfitting in Token Selection of Attention Mechanism](https://arxiv.org/abs/2409.17625)

### Requirements
- Python
- PyTorch
- Wandb

To install requirements:
```bash
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

If you are first to [Weights & Biases](https://wandb.ai/site), please login:
```bash
wandb login
```

### Training
To train the model in the same setting as our paper, run this command:
```python
python main.py
```

To run with different hyperparameters, you can either specify them in the command line or modify `config/main.yaml`.

Additionally, you can run with multiple parameters as:
```python
python main.py --multirun signal_norm=5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100
```

### Reproducing Figures
Run code blocks in `plot.ipynb` step by step.

For some of the figures, you need to run the main file with `multirun` option.

### Citation
If you find our work useful for your research, please cite using this BibTeX:
```BibTeX
@article{sakamoto2024benign,
      title={Benign or not-benign overfitting in token selection of attention mechanism}, 
      author={Sakamoto, Keitaro and Sato, Issei},
      journal={arXiv preprint arXiv:2409.17625},
      year={2024}
}
```
