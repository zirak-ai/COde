# Casangels Oro-dental Dataset (COde)

This dataset is associated with our paper, currently under peer review, titled:  
**"A benchmark multimodal oro-dental dataset for large vision-language models."**

In this repository, we have shared only the textual part of the dataset.  
The complete dataset is available at:  
[https://doi.org/10.57967/hf/6421](https://doi.org/10.57967/hf/6421)

## Data Preprocessing

The data preprocessing pipeline used to prepare the dataset is implemented in [`data-preprocessing.ipynb`](./data-preprocessing.ipynb). This notebook includes the steps for cleaning, formatting, and organizing the raw data before training. If you would like more details about the preprocessing workflow, please refer to this code file.

## Supervised Finetuning

The configuration and setup for supervised finetuning are provided in [`SFT-Config.yaml`](./SFT-Config.yaml). This file contains the training parameters, model settings, and finetuning configuration used in our experiments. For more information, please refer to this configuration file.

## Evaluation

The evaluation procedure for the finetuned model is implemented in [`evaluate.py`](./evaluate.py). This script contains the metrics and testing pipeline used to assess model performance on the benchmark dataset.
