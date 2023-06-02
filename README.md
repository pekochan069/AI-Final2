# AI Final

## Project Setup

### Install requirements

Requires **Conda**

Highly recommend to install Conda from [miniforge](https://github.com/conda-forge/miniforge) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)

After installing conda, create environment and activate it

```shell
conda env create -f environment.yaml
conda activate torch
```

## remove huggingface dataset (DIV2K)

```bash
rm -rf ~/.cache/huggingface
```
