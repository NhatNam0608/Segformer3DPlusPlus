import os
import json
import wandb
import torch
from tqdm import tqdm
from typing import Dict
from termcolor import colored
from torch.utils.data import DataLoader
from metrics.segmentation_metrics import SlidingWindowInference


#################################################################################################
class Segmentation_Trainer:
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_scheduler: torch.optim.lr_scheduler.LRScheduler,
        training_scheduler: torch.optim.lr_scheduler.LRScheduler,
        accelerator=None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None
        self.accelerator = accelerator
        self.wandb_tracker = accelerator.get_tracker("wandb")

        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.resume = self.config['wandb']['parameters']['resume']
        self.run = wandb.init(entity=config['wandb']['parameters']['entity'], project=config['wandb']['parameters']['project'])
        self.start_epoch = 0
        self.best_val_dice = 0.0 
        self.current_epoch = 0
        if self.resume:
            self._load_checkpoint()
            self._load_best_checkpoint()
            
        # external metric functions we can add
        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"],
        )
    def _save_checkpoint(self) -> None:
        output_dir = "checkpoint"
        os.makedirs(output_dir, exist_ok=True)
        # Save model, optimizer, scheduler, etc.
        self.accelerator.save_state(output_dir)
        # Save additional metadata (e.g., epoch)
        meta = {
            "epoch": self.current_epoch,
        }
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
    
        # Upload to WandB as artifact
        artifact = wandb.Artifact(output_dir, type="model")
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)
    def _save_best_checkpoint(self) -> None:
        output_dir = "best-checkpoint"
        os.makedirs(output_dir, exist_ok=True)
        # Save model, optimizer, scheduler, etc.
        self.accelerator.save_state(output_dir)
        # Save additional metadata (e.g., epoch)
        meta = {
            "best_val_dice": self.best_val_dice,
        }
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
    
        # Upload to WandB as artifact
        artifact = wandb.Artifact(output_dir, type="model")
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)
    def _load_checkpoint(self) -> None:
        artifact = self.run.use_artifact(f'{self.config["wandb"]["parameters"]["entity"]}/{self.config["wandb"]["parameters"]["project"]}/checkpoint:latest', type='model')
        artifact_dir = artifact.download()
        self.accelerator.load_state(artifact_dir)
        # Load metadata
        meta_path = os.path.join(artifact_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.start_epoch = int(meta.get("epoch", 0)) + 1
    def _load_best_checkpoint(self) -> None:
        artifact = self.run.use_artifact(f'{self.config["wandb"]["parameters"]["entity"]}/{self.config["wandb"]["parameters"]["project"]}/best-checkpoint:latest', type='model')
        artifact_dir = artifact.download()
        # Load metadata
        meta_path = os.path.join(artifact_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.best_val_dice = float(meta.get("best_val_dice", 0))
        
    def _train_step(self) -> float:
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
        self.model.train()

        for index, raw_data in enumerate(self.train_dataloader):
            # add in gradient accumulation
            with self.accelerator.accumulate(self.model):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # zero out existing gradients
                self.optimizer.zero_grad()

                # forward pass
                predicted = self.model.forward(data)
                # calculate loss
                loss = self.criterion(predicted, labels)

                # backward pass
                self.accelerator.backward(loss)

                # update gradients
                self.optimizer.step()

                # update loss
                epoch_avg_loss += loss.item()
                wandb.log({"train_loss": loss.item()})
                if self.print_every:
                    if index % self.print_every == 0:
                        self.accelerator.print(
                            f"epoch: {str(self.current_epoch).zfill(4)} -- "
                            f"train loss: {(epoch_avg_loss / (index + 1)):.5f} -- "
                            f"lr: {self.scheduler.get_last_lr()[0]}"
                        )

        epoch_avg_loss = epoch_avg_loss / (index + 1)
        return epoch_avg_loss

    def _val_step(self) -> float:
        total_dice = 0.0
        self.model.eval()

        with torch.no_grad():
            for index, (raw_data) in enumerate(self.val_dataloader):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # calculate metrics
                if self.calculate_metrics:
                    mean_dice = self._calc_dice_metric(data, labels)
                    total_dice += mean_dice

        avg_dice = total_dice / float(index + 1)
        return avg_dice

    def _calc_dice_metric(self, data, labels) -> float:
        """_summary_
        Args:
            predicted (_type_): _description_
            labels (_type_): _description_

        Returns:
            float: _description_
        """
        avg_dice_score = self.sliding_window_inference(
            data,
            labels,
            self.model,
        )
        return avg_dice_score

    def _run_train_val(self) -> None:
        """_summary_"""
        # Tell wandb to watch the model and optimizer values
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model, self.criterion, log="all", log_freq=10, log_graph=True
            )

        # Run Training and Validation
        for epoch in tqdm(range(self.start_epoch, self.num_epochs)):
            # update epoch
            self.current_epoch = epoch
            self._update_scheduler()

            # run a single training step
            train_loss = self._train_step()
            # run a single validation step
            avg_dice = self._val_step()

            # update metrics
            self._update_metrics(avg_dice)


            # save and print
            self._save_and_print(avg_dice, train_loss)

            # update schduler
            self.scheduler.step()

    def _update_scheduler(self) -> None:
        """_summary_"""
        if self.warmup_enabled:
            if self.current_epoch == 0:
                self.accelerator.print(
                    colored(f"\n[info] -- warming up learning rate \n", color="red")
                )
                self.scheduler = self.warmup_scheduler
            elif self.current_epoch > 0 and self.current_epoch < self.warmup_epochs:
                if self.scheduler is None:
                    self.scheduler = self.warmup_scheduler
            elif self.current_epoch == self.warmup_epochs:
                self.accelerator.print(
                    colored(
                        f"\n[info] -- switching to learning rate decay schedule \n",
                        color="red",
                    )
                )
                self.scheduler = self.training_scheduler
            elif self.current_epoch > self.warmup_epochs:
                if self.scheduler is None:
                    self.scheduler = self.training_scheduler
                    
        elif self.scheduler is None:
            self.accelerator.print(
                colored(
                    f"\n[info] -- setting learning rate decay schedule \n",
                    color="red",
                )
            )
            self.scheduler = self.training_scheduler

    def _update_metrics(self, avg_dice) -> None:
        if self.calculate_metrics:
            if avg_dice >= self.best_val_dice:
                self.best_val_dice = avg_dice

    def _save_and_print(self, avg_dice, train_loss) -> None:
        """_summary_"""
        # print only on the first gpu
        if avg_dice >= self.best_val_dice:
            # save checkpoint and log
            self._save_best_checkpoint()
            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{train_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"val mean_dice -- {colored(f'{avg_dice:.5f}', color='green')} -- saved"
            )
        else:
            self.accelerator.print(
                f"epoch -- {str(self.current_epoch).zfill(4)} || "
                f"train loss -- {train_loss:.5f} || "
                f"lr -- {self.scheduler.get_last_lr()[0]:.8f} || "
                f"val mean_dice -- {avg_dice:.5f}"
            )
        self._save_checkpoint()
    def train(self) -> None:
        """
        Runs a full training and validation of the dataset.
        """
        self._run_train_val()
        self.accelerator.end_training()

    def evaluate(self) -> None:
        val_dice = self._val_step()
        print(f'''Mean dice on evaluation dataset is {val_dice}''')
