import numpy as np
from typing import Dict
from accelerate import Accelerator
from evaluators.evaluators import Segmentation_Evaluator
from architectures.build_architecture import build_architecture
from dataloaders.build_datasets import build_dataset, build_dataloader
from utils.utils import seed_everything, load_config

##################################################################################################
def eval(config) -> Dict:
    # set seed
    seed_everything(config['training_parameters']['seed'])

    # build validation dataset & validataion data loader
    valset = build_dataset(
        name=config["dataset"]["name"],
        dataset_args=config["dataset"]["val_dataset"],
    )
    val_dataloader = build_dataloader(
        dataset=valset,
        dataloader_args=config["dataset"]["val_dataloader"],
    )

    # build the Model
    model = build_architecture(config)

    # use accelarate
    accelerator = Accelerator()

    # convert all components to accelerate
    model = accelerator.prepare_model(model=model)
    val_dataloader = accelerator.prepare_data_loader(data_loader=val_dataloader)

    # set up trainer
    evaluator = Segmentation_Evaluator(
        config=config,
        model=model,
        val_dataloader=val_dataloader,
        accelerator=accelerator
    )

    # run train
    evaluator.evaluate()
if __name__ == "__main__":
    # load config
    config = load_config("config.yaml")
    eval(config)