from typing import Dict
from termcolor import colored
from accelerate import Accelerator

from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler
from losses.losses import build_loss_fn
from trainers.trainers import Segmentation_Trainer
from architectures.build_architecture import build_architecture
from dataloaders.build_datasets import build_dataset, build_dataloader
from utils.utils import load_config, seed_everything

##################################################################################################
def display_info(config, accelerator, trainset, valset, model):
    # print experiment info
    accelerator.print(f"-------------------------------------------------------")
    accelerator.print(f"[info]: Experiment Info")
    accelerator.print(
        f"[info] ----- Project: {colored(config['wandb']['parameters']['project'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Resume: {colored(config['wandb']['parameters']['resume'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Mode: {colored(config['wandb']['parameters']['mode'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Run: {colored(config['wandb']['parameters']['name'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Notes: {colored(config['wandb']['parameters']['notes'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Batch Size for Training: {colored(config['dataset']['train_dataloader']['batch_size'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Num Epochs for Training: {colored(config['training_parameters']['num_epochs'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Loss: {colored(config['loss_fn']['loss_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Optimizer: {colored(config['optimizer']['optimizer_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Train Dataset Size: {colored(len(trainset), color='red')}"
    )
    accelerator.print(
        f"[info] ----- Test Dataset Size: {colored(len(valset), color='red')}"
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"[info] ----- Num Clases: {colored(config['model']['parameters']['num_classes'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Params: {colored(pytorch_total_params, color='red')}"
    )
    accelerator.print(f"-------------------------------------------------------")

##################################################################################################
def train(config) -> Dict:
    """
    Builds training
    Args:
        config (Dict): configuration
    """
    # set seed
    seed_everything(config['training_parameters']['seed'])

    # build training dataset & training data loader
    trainset = build_dataset(
        name=config["dataset"]["name"],
        dataset_args=config["dataset"]["train_dataset"],
    )
    trainloader = build_dataloader(
        dataset=trainset,
        dataloader_args=config["dataset"]["train_dataloader"],
    )

    # build validation dataset & validataion data loader
    valset = build_dataset(
        name=config["dataset"]["name"],
        dataset_args=config["dataset"]["val_dataset"],
    )
    valloader = build_dataloader(
        dataset=valset,
        dataloader_args=config["dataset"]["val_dataloader"],
    )

    # build the Model
    model = build_architecture(config)

    # set up the loss function
    criterion = build_loss_fn(
        loss_type=config["loss_fn"]["loss_type"],
        loss_args=config["loss_fn"]["loss_args"],
    )

    # set up the optimizer
    optimizer = build_optimizer(
        model=model,
        optimizer_type=config["optimizer"]["optimizer_type"],
        optimizer_args=config["optimizer"]["optimizer_args"],
    )

    # set up schedulers
    warmup_scheduler = build_scheduler(
        optimizer=optimizer, scheduler_type="warmup_scheduler", config=config
    )
    training_scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="training_scheduler",
        config=config,
    )

    # use accelarate
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=config["training_parameters"]["grad_accumulate_steps"],
    )
    accelerator.init_trackers(
        project_name=config['wandb']['parameters']["project"]
    )

    # display experiment info
    display_info(config, accelerator, trainset, valset, model)

    # convert all components to accelerate
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    trainloader = accelerator.prepare_data_loader(data_loader=trainloader)
    valloader = accelerator.prepare_data_loader(data_loader=valloader)
    warmup_scheduler = accelerator.prepare_scheduler(scheduler=warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(scheduler=training_scheduler)

    # create a single dict to hold all parameters
    storage = {
        "model": model,
        "trainloader": trainloader,
        "valloader": valloader,
        "criterion": criterion,
        "optimizer": optimizer,
        "warmup_scheduler": warmup_scheduler,
        "training_scheduler": training_scheduler,
    }

    # set up trainer
    trainer = Segmentation_Trainer(
        config=config,
        model=storage["model"],
        optimizer=storage["optimizer"],
        criterion=storage["criterion"],
        train_dataloader=storage["trainloader"],
        val_dataloader=storage["valloader"],
        warmup_scheduler=storage["warmup_scheduler"],
        training_scheduler=storage["training_scheduler"],
        accelerator=accelerator,
    )

    # run train
    trainer.train()