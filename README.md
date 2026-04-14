# PyTorch Template for DL projects with HuggingFace Integration

<p align="center">
  <a href="#about">About</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/CITATION.cff">
   <img src="https://img.shields.io/badge/cite-this%20repo-purple">
</a>
</p>

## About

This repository contains a template for [PyTorch](https://pytorch.org/)-based Deep Learning projects with [HuggingFace](https://huggingface.co/) Integration.

It is designed for researchers and engineers who appreciate the simplicity of the `Trainer` from [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for large-scale training, but need a more **general, customizable training framework** that goes beyond NLP use cases.

By combining [PyTorch](https://pytorch.org/), [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index), and [Hydra](https://hydra.cc/docs/intro/), this template enables scalable training, clean configuration management, and reproducible experimentation without sacrificing flexibility.

Core features:

1. **Scalable Training** (via [Accelerate](https://huggingface.co/docs/accelerate/index)):
   - Multi-GPU and multi-node training
   - Distributed inference support
   - Gradient accumulation
   - Automatic mixed precision (AMP)

2. [HuggingFace](https://huggingface.co/) Ecosystem Compatibility:
   - Checkpoints are fully compatible with [Hugging Face Transformers](https://huggingface.co/docs/transformers/index).
   - Automatic pushing of checkpoints / models to the [Hugging Face Hub](https://huggingface.co/docs/hub/index).
   - Models inherit from [transformers](https://huggingface.co/docs/transformers/index) `PreTrainedModel` and configs from `PreTrainedConfig`.
   - Models can be loaded with `AutoModel.from_pretrained()`.
   - Model implementations remain compatible with the broader Transformers ecosystem.

3. [Hydra](https://hydra.cc/docs/intro/)-based configuration for enhanced control:
   - Structured and composable configs.
   - Easy re-configuration via CLI.

As in the [main branch](https://github.com/Blinorot/pytorch_project_template/tree/main), the template utilizes different python-dev techniques to improve code readability. Configuration methods enhance reproducibility and experiments control.

> [!IMPORTANT]
> This branch is a **general template**. So, some parts of the code, e.g. dataset and metrics, are filled with dummy examples, showing how the code works and the expected flow. We advise you to read the doc-strings for all functions. The final users can add code required for their own tasks.

## Tutorials

This template utilizes experiment tracking techniques, such as [WandB](https://docs.wandb.ai/) and [Comet ML](https://www.comet.com/docs/v2/), [Hydra](https://hydra.cc/docs/intro/) for the configuration, and [transformers](https://huggingface.co/docs/transformers/index) with [accelerate](https://huggingface.co/docs/accelerate/index) for the integration of [HuggingFace](https://huggingface.co/) infrastructure and large-scale training. It also automatically reformats code and conducts several checks via [pre-commit](https://pre-commit.com/). If you are not familiar with these tools, we advise you to look at the tutorials below:

- [Python Dev Tips](https://github.com/ebezzam/python-dev-tips): information about [Git](https://git-scm.com/doc), [pre-commit](https://pre-commit.com/), [Hydra](https://hydra.cc/docs/intro/), and other stuff for better Python code development. The YouTube recording of the workshop is available [here](https://youtu.be/okxaTuBdDuY).

- [Seminar on R&D Coding 2025](https://youtu.be/PE1zaW5it_A): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with discussion on logging, project-based coding, configuration, and reproducibility. The materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/tree/summer25/day05).

- [Seminar on R&D Coding 2024](https://youtu.be/sEA-Js5ZHxU): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with template discussion and reasoning. It also explains how to work with [WandB](https://docs.wandb.ai/). The seminar materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/blob/main/day03/Seminar_WandB_and_Coding.ipynb).

- [HSE DLA Course Introduction Week](https://github.com/markovka17/dla/tree/2024/week01): combines the two seminars above into one with some updates, including an extra example for [Comet ML](https://www.comet.com/docs/v2/).

- [PyTorch Basics](https://github.com/markovka17/dla/tree/2024/week01/intro_to_pytorch): several notebooks with [PyTorch](https://pytorch.org/docs/stable/index.html) basics and corresponding seminar recordings from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/).

- [Accelerate Quicktour](https://huggingface.co/docs/accelerate/quicktour): explanation of the core differences between classic training and `accelerate`-training pipelines. You can also check the tutorials on the same page.

- [Transformers Quicktour](https://huggingface.co/docs/transformers/quicktour): explanation of the `transformer` basics and what `PreTrainedModel` allows to do.

To start working with a template, just click on the `use this template` button.

<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>

You can choose any of the branches as a starting point. [Set your choice as the default branch](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch) in the repository settings. You can also [delete unnecessary branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).

## Installation

Installation may depend on your task. The general steps are the following:

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
   train.py -cn=HYDRA_CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `ACCELERATE_CONFIG` is a config from `src/configs/accelerate`, `HYDRA_CONFIG_NAME` is a config from `src/configs`, and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

Example for a single-GPU training:

```bash
accelerate launch --config-file src/configs/accelerate/single.yaml \
   train.py -cn=HYDRA_CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Example for a multi-GPU training:

```bash
accelerate launch --config-file src/configs/accelerate/multigpu.yaml \
   train.py -cn=HYDRA_CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

You can also change the number of GPUs or the precision used by `--num-processes NUM_GPUS` and `--mixed-precision PRECISION_TYPE` after choosing `accelerate` config and before `train.py`. See [accelerate documentation](https://huggingface.co/docs/accelerate/index).

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
    writer.run_name="multinode-training"
'
```

> [!IMPORTANT]
> We assume that all nodes have a shared filesystem. Therefore, the model checkpoints are saved only by the main process on the head node. If your filesystem is not shared, you need to replace `accelerator.is_main_process` if-else statements with `accelerator.is_local_main_process` in saving-related parts of the code.

To run inference (evaluate the model or save predictions):

```bash
accelerate launch --config-file ACCELERATE_CONFIG inference.py \
   inferencer.from_pretrained=PATH_TO_MODEL_WEIGHTS \
   HYDRA_CONFIG_ARGUMENTS
```

Where `PATH_TO_MODEL_WEIGHTS` is the path to the saved pretrained model, e.g., `saved/testing/checkpoint-best/model_weights`.

## Useful Links:

You may find the following links useful:

- [Report branch](https://github.com/Blinorot/pytorch_project_template/tree/report): Guidelines for writing a scientific report/paper (with an emphasis on DL projects).

- [CLAIRE Template](https://github.com/CLAIRE-Labo/python-ml-research-template): additional template by [EPFL CLAIRE Laboratory](https://www.epfl.ch/labs/claire/) that can be combined with ours to enhance experiments reproducibility via [Docker](https://www.docker.com/).

- [Mamba](https://github.com/mamba-org/mamba) and [Poetry](https://python-poetry.org/): alternatives to [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) and [pip](https://pip.pypa.io/en/stable/installation/) package managers given above.

- [Awesome README](https://github.com/matiassingers/awesome-readme): a list of awesome README files for inspiration. Check the basics [here](https://github.com/PurpleBooth/a-good-readme-template).

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
