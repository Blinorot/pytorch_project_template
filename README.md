# Image classification with PyTorch and HuggingFace

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This in an example usage of a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template). The task is to do image classification. We use [MNIST](https://yann.lecun.com/exdb/mnist/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.

This branch uses the [HuggingFace Integration Variant](https://github.com/Blinorot/pytorch_project_template/tree/hf_main) of the template with multi-GPU and multi-node training support.

## Installation

Follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
accelerate launch --config-file ACCELERATE_CONFIG \
   train.py -cn=CONFIG_NAME \
   HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
accelerate launch --config-file ACCELERATE_CONFIG \
   inference.py HYDRA_CONFIG_ARGUMENTS
```

## Examples

To train a simple MLP on MNIST, run:

```bash
accelerate launch --config-file src/configs/accelerate/single.yaml \
   train.py model=baseline
```

If you want train your MLP on CIFAR-10, run this instead:

```bash
accelerate launch --config-file src/configs/accelerate/single.yaml \
   train.py model=baseline \
   model.n_feats=3072 \
   datasets=cifar \
   transforms/batch_transforms=cifar
```

If you want to fine-tune ResNet18 on CIFAR-10, run this:

```bash
accelerate launch --config-file src/configs/accelerate/single.yaml \
   train.py model=resnet \
   datasets=cifar \
   model.input_channels=3 \
   transforms/batch_transforms=cifar_resnet
```

Replace `single.yaml` with `multigpu.yaml` for multi-GPU training.

For a multi-node setup, we provide an example for a [SLURM](https://github.com/schedmd/slurm)-based server:

```bash
#!/bin/bash
#SBATCH --job-name=multinode_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=<PARTITION_NAME>
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29510

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

srun \
  --ntasks=${SLURM_JOB_NUM_NODES} \
  --ntasks-per-node=1 \
  --container-writable \
  --environment=<CONTAINER_ENV_NAME> \
  bash -lc '
set -euo pipefail

echo "HOSTNAME=$(hostname)"
echo "SLURM_NODEID=${SLURM_NODEID}"

# Uncomment if the container does not have all the packages
# python -m pip install -r requirements.txt

# Adjust num-processes = GPUs per node × number of nodes

accelerate launch \
  --config-file src/configs/accelerate/multigpu.yaml \
  --num-machines ${SLURM_JOB_NUM_NODES} \
  --num-processes 8 \
  --machine-rank ${SLURM_NODEID} \
  --main-process-ip '"$MASTER_ADDR"' \
  --main-process-port '"$MASTER_PORT"' \
  train.py \
    writer.run_name="multinode-training" \
    model=resnet datasets=cifar \
    model.input_channels=3 \
    transforms/batch_transforms=cifar_resnet
'
```

For inference, call the `inference.py` script with a path to checkpoint:

```bash
accelerate launch --config-file src/configs/accelerate/single.yaml \
   inference.py \
   inferencer.from_pretrained=PATH_TO_MODEL_WEIGHTS \
   model=resnet \
   datasets=cifar_test \
   model.input_channels=3 \
   transforms/batch_transforms=cifar_resnet
```

Where `PATH_TO_MODEL_WEIGHTS` is the path to the saved pretrained model, e.g., `saved/testing/checkpoint-best/model_weights`.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
