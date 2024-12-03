# ARRG prototype model

Code for Paper "[CID at RRG24: Attempting in a Conditionally Initiated Decoding of Radiology Report Generation with Clinical Entities](https://aclanthology.org/2024.bionlp-1.49/)"

A model for the Shared task on Large-Scale Radiology Report Generation @ BioNLP ACL'24

More information can be found at https://stanford-aimi.github.io/RRG24/

Our pre-trained models are available [here](https://drive.google.com/drive/folders/1dAXc8EpQ36g0cWtwP5qM0yj_KYHtLEIo?usp=drive_link)

## Install

- conda create --name arrg_prototype python=3.8 -y
- conda activate arrg_prototype

- pip install vilmedic
- pip install datasets
- pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
- pip install --upgrade transformers
- pip install setuptools==59.5.0
- pip install tensorboard==2.11

(Our env: CUDA/11.7, transformers==4.41.1, torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2, datasets=2.19.1, vilmedic==1.3.3)

### Check the code

In terminal, type:
```
$ python
>>> import vilmedic
```
and you might see errors. We will address them in the next part.

Once you see this message
```
>>> timestamp WARNING: Language en package default expects mwt, which has been added
```
it means the env is ready.

### Errors

- ImportError: cannot import name 'ViTImageProcessor' from 'transformers' (`...envs/arrg_prototype/lib/python3.8/site-packages/transformers/__init__.py`)
  - vilmedic installs Transformers==4.23.1 by default, which does not have ViTImageProcessor. It was introduced since 4.25.1
  - `pip install --upgrade transformers`
- ModuleNotFoundError: No module named 'transformers.generation_beam_constraints'` or others modules e.g. `transformers.generation_xxx
  - This is because `transformers.generation_beam_xxx` has been renamed to `transformers.generation.xxx` since [4.25.1](https://github.com/huggingface/transformers/tree/v4.40.0/src/transformers/generation)
  - Go to the error line (should be in `...envs/arrg_prototype/lib/python3.8/site-packages/vilmedic/blocks/huggingface/decoder/beam_search.py`)
  - Change the module from `transformers.generation_xxx` to `transformers.generation.xxx`
- ImportError: cannot import name 'torch_int_div' from 'transformers.pytorch_utils' (`...envs/arrg_prototype/lib/python3.8/site-packages/transformers/pytorch_utils.py`)
  - This function has been removed since transformers==4.28.0 (check in [4.23.1](https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/pytorch_utils.py#L35))
  - Go to the error file (`...envs/arrg_prototype/lib/python3.8/site-packages/vilmedic/blocks/huggingface/decoder/beam_search.py`)
  - Comment out this import

    ```python
    # from transformers.pytorch_utils import torch_int_div
    ```
  - Copy the following code into the error file 

    ```python
    from packaging import version
    parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
    is_torch_less_than_1_8 = parsed_torch_version_base < version.parse("1.8.0")
    def torch_int_div(tensor1, tensor2):
        """
        A function that performs integer division across different versions of PyTorch.
        """
        if is_torch_less_than_1_8:
            return tensor1 // tensor2
        else:
            return torch.div(tensor1, tensor2, rounding_mode="floor")
    ```
- RuntimeError: CUDA error: no kernel image is available for execution on the device\n CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
  - vilmedic installs torch==1.10.1 by default.
  - Check the cuda version, and install a [correct torch version](https://pytorch.org/get-started/previous-versions/).
    - `nvcc --version`
- TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'
  - Use `pip install tensorboard==2.11` (2.14 will cause this error)
- AttributeError: module 'distutils' has no attribute 'version'
  - Solution from this [link](https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version)
  - `pip install setuptools==59.5.0`
- You may also see the following errors, which is fine:
  ```
  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
  vilmedic 1.3.3 requires torch==1.10.1, but you have torch 2.0.1 which is incompatible.
  vilmedic 1.3.3 requires torchvision==0.11.2, but you have torchvision 0.15.2 which is incompatible.
  vilmedic 1.3.3 requires transformers==4.23.1, but you have transformers 4.41.1 which is incompatible.
  ```

## Prepare datasets

Datasets download instruction: https://stanford-aimi.github.io/RRG24/

Please also download our pre-constructed graph labels from this [link](https://drive.google.com/drive/folders/1dAXc8EpQ36g0cWtwP5qM0yj_KYHtLEIo?usp=sharing).

Once you download the datasets and the label files, you can follow the script `./preprocessing/0_prepare_datasets.ipynb` to prepare the datasets for training

## Model training

Use `./train.py` to train the model. Training config is in `./config.py`

## Evaluation

The evaluation metrics and results can be found at: https://vilmedic.app/misc/bionlp24/leaderboard
