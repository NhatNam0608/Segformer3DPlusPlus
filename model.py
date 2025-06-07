import torch
import numpy as np
from utils.utils import load_config
from architectures.build_architecture import build_architecture
if __name__ == "__main__":
    config = load_config("config.yaml")

    # Initialize the model
    model = build_architecture(config)
    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
   
