import sys
import os
from typing import Dict
from monai.data import DataLoader
import torch
from augmentations.augmentations import build_augmentations
from dataloaders.brats2021_seg import Brats2021SegDataset
from dataloaders.acdc import ACDCDataset
from utils.utils import load_config, generate_gif

def build_dataset(name: str, dataset_args: Dict):
    if name == "brats2021_seg":
        if dataset_args['train']:
            root_dir=os.path.join(dataset_args["root"],'train')
        else:
            root_dir=os.path.join(dataset_args["root"],'val')
        dataset = Brats2021SegDataset(
                root_dir=root_dir,
                transform=build_augmentations(dataset_args['train']),
        )
        return dataset
    elif name == 'acdc':
        print("Build ACDC dataset")
        if dataset_args['train']:
            root_dir=os.path.join(dataset_args["root"],'train')
        else:
            root_dir=os.path.join(dataset_args["root"],'val')
        dataset = ACDCDataset(
                root_dir=root_dir,
                transform=build_augmentations(dataset_args['train']),
        )
        return dataset
    elif name == 'synapse':
        pass
    else:
        raise ValueError(
            f'{name} not valid'
        )


######################################################################
def build_dataloader(
    dataset, dataloader_args: Dict
) -> DataLoader:
    """builds the dataloader for given dataset
    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
    Returns:
        DataLoader: _description_
    """
    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=dataloader_args["num_workers"],
        drop_last=dataloader_args["drop_last"],
        pin_memory=pin_memory,
    )
    return dataloader

 
if __name__ == '__main__':
    import time
    config = load_config('config.yaml')  
    trainset = build_dataset(
        name=config["dataset"]["name"],
        dataset_args=config["dataset"]["train_dataset"],
    )
    data = trainset[1]
    generate_gif(data[0]['image'][0].cpu().numpy(),"BraTS2021_0.gif") 
    generate_gif(data[1]['image'][0].cpu().numpy(),"BraTS2021_1.gif") 
    generate_gif(data[2]['image'][0].cpu().numpy(),"BraTS2021_2.gif") 
    generate_gif(data[3]['image'][0].cpu().numpy(),"BraTS2021_3.gif") 
    trainloader = build_dataloader(
        dataset=trainset,
        dataloader_args=config["dataset"]["train_dataloader"],
    )
    start = time.time()
    for i, data in enumerate(trainloader):
        pass
        # images = data['image']
        # labels = data['label']
        # print(images.shape)
        # print(labels.shape)
        # break
    end = time.time()
    print(end - start)
