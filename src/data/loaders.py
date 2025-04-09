import torch
from torch.utils.data.dataset import Dataset
from os import path as osp
from datasets import eth3d_dataset
from functools import partial
from transforms import collate_fn_descriptor, calibrate_neighbors

import pytorch_lightning as pl
from torch.utils.data import (
    Dataset, DataLoader, random_split
)

class eth3d_DataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, config):
        super().__init__()
        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': config.batch_size,
            'num_workers': config.num_workers
        }

        self.val_loader_params = {
            'batch_size': config.batch_size,
            'num_workers': config.num_workers
        }

        self.test_loader_params = {
            'batch_size': config.batch_size,
            'num_workers': config.num_workers
        }

    def setup(self, stage=None):
        full_dataset = eth3d_dataset(self.data_dir)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        if stage == "test" or stage is None:
            self.test_dataset = eth3d_dataset(self.data_dir, train=False)

    def train_dataloader(self, config):
        return DataLoader(self.train_dataset, 
                          collate_fn=partial(collate_fn_descriptor, config=config.dataset, neighborhood_limits=neighborhood_limits),
                          **self.train_loader_params)

    def val_dataloader(self, config):
        return DataLoader(self.val_dataset, 
                          collate_fn=partial(collate_fn_descriptor, config=config.dataset, neighborhood_limits=neighborhood_limits),
                          **self.val_loader_params)

    def test_dataloader(self, config):
        return DataLoader(self.test_dataset, 
                          collate_fn=partial(collate_fn_descriptor, config=config.dataset, neighborhood_limits=neighborhood_limits),
                          **self.test_loader_params)