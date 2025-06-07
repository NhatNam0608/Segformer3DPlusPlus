import torch
import torch.nn as nn
from safetensors.torch import load_file

from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.transforms import Activations
from monai.inferers import sliding_window_inference
from architectures.build_architecture import build_architecture
from utils.utils import load_config, create_gif_from_volume, seed_everything, load_modalities

class SlidingWindowInference_:
    def __init__(self, roi: tuple, sw_batch_size: int):
        self.post_transform = Compose([
            Activations(sigmoid=True),
            AsDiscrete(argmax=False, threshold=0.5),
        ])
        self.sw_batch_size = sw_batch_size
        self.roi = roi

    def _infer_output(self, val_inputs: torch.Tensor, model: nn.Module):
        logits = sliding_window_inference(
            inputs=val_inputs,
            roi_size=self.roi,
            sw_batch_size=self.sw_batch_size,
            predictor=model,
            overlap=0.5,
        )
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [
            self.post_transform(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]
        return torch.stack(val_output_convert)  # (B, C, H, W, D)

def inference(config, image: str, cpkt:str):
    # Set seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(config['training_parameters']['seed'])

    # Load input tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()
    image_tensor = image_tensor.to(device)

    # Load model
    model = build_architecture(config)
    weights_path = cpkt
    state_dict = load_file(weights_path, device=str(device))
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Inference using MONAI sliding window
    with torch.no_grad():
        output_tensor = SlidingWindowInference_(
            roi=config["sliding_window_inference"]["roi"],
            sw_batch_size=config["sliding_window_inference"]["sw_batch_size"],
        )._infer_output(image_tensor, model)  # We call internal inference logic

    return output_tensor.cpu().numpy() 

if __name__ == "__main__":
    image_pth = "data/brats2021_seg/processed/val/BraTS2021_01657/BraTS2021_01657_modalities.pt"
    label_pth = "data/brats2021_seg/processed/val/BraTS2021_01657/BraTS2021_01657_label.pt"
    image = load_modalities(image_pth)
    label = load_modalities(label_pth).astype(int)
    config = load_config("config.yaml")
    prediction = inference(config, image, cpkt="checkpoint/segformer3d-epa-brats/model.safetensors")[0]
    create_gif_from_volume(volume=image, prediction=label, gif_path="assets/brats2021_seg/volume_with_label/BraTS2021_01657.gif")
    create_gif_from_volume(volume=image, prediction=prediction, gif_path="assets/brats2021_seg/volume_with_prediction/BraTS2021_01657.gif")
