import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    config = load_config(args.config)

    # model and data_module
    model = (config) 
    data_module = (config)

    # tools
    logger = TensorBoardLogger(save_dir='experiments/tensorboard_logs', name='PICNet_log', default_hp_metric=False)    # TensorBoard Logger
    profiler = SimpleProfiler()
    
    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        profiler=profiler
    )
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()