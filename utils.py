import logging
import os
import random

import numpy as np
import torch

LOGGER = logging.getLogger("utils")


class StableRandomSampler:
    def __init__(self, data_source, num_epoch):
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.seeds = iter(torch.empty((num_epoch), dtype=torch.int64).random_().tolist())

    def __iter__(self):
        seed = next(self.seeds)
        generator = torch.Generator()
        generator.manual_seed(seed)
        rand_list = torch.randperm(self.num_samples, generator=generator).tolist()
        LOGGER.debug("Shuffle batches with seed = %d", seed)

        yield from rand_list

    def __len__(self) -> int:
        return self.num_samples

    def resume_to_epoch(self, epoch):
        # To make sure the reproducibility. resume_to_epoch means the index of epoch to be resumed.
        if epoch > 0:
            seeds = [next(self.seeds) for i in range(epoch)]
            LOGGER.debug("Skip %d seeds = %s", epoch, seeds)


def set_seed(seed):
    random.seed(seed)  # Python的随机性
    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
