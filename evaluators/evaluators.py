import wandb
import torch    
import numpy as np          
from typing import Dict
from torch.utils.data import DataLoader
from metrics.segmentation_metrics import SlidingWindowInference


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
        print("TC:", mean_per_class[0])
        print("WT:", mean_per_class[1])
        print("ET:", mean_per_class[2])
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
   
