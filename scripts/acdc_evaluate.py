import wandb
import torch    
import numpy as np          
from typing import Dict
from accelerate import Accelerator
from metrics.segmentation_metrics import SlidingWindowInference
from architectures.build_architecture import build_architecture
from dataloaders.build_datasets import build_dataset, build_dataloader
from utils.utils import seed_everything, load_config

from torch.utils.data import DataLoader


#################################################################################################
class Segmentation_Evaluator:
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        val_dataloader: DataLoader,
        accelerator = None
    ) -> None:
        self.config = config
        self.model = model
        self.accelerator = accelerator  
        self.val_dataloader = val_dataloader
        self.sliding_window_inference = SlidingWindowInference(
            config["sliding_window_inference"]["roi"],
            config["sliding_window_inference"]["sw_batch_size"],
        )
        self._load_best_checkpoint()

    def _load_best_checkpoint(self) -> None:
        self.accelerator.load_state(f"checkpoint/{self.config['model']['name']}")
    
    def evaluate(self) -> float:
        self.model.eval()
        acc_list = []
        with torch.no_grad():
            for index, (raw_data) in enumerate(self.val_dataloader):
                # get data ex: (data, target)
                data, labels = (
                    raw_data["image"],
                    raw_data["label"],
                )
                # calculate metrics
                acc = self._calc_dice_metric(data, labels)
                acc_list.append(acc)
        acc_array = np.array(acc_list)
        mean_per_class = np.mean(acc_array, axis=0)  
        print("RV:", mean_per_class[0])
        print("Myo:", mean_per_class[1])
        print("LV:", mean_per_class[2])
        print("mean dice:", np.mean(mean_per_class)*100)
        

    def _calc_dice_metric(self, data, labels) -> float:
        """_summary_
        Args:
            predicted (_type_): _description_
            labels (_type_): _description_

        Returns:
            float: _description_
        """
        acc = self.sliding_window_inference(
            data,
            labels,
            self.model,
        )
        return acc

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