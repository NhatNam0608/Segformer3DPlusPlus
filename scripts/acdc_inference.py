import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from safetensors.torch import load_file

from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.transforms import Activations
from monai.inferers import sliding_window_inference
from PIL import Image
import os
from utils.utils import seed_everything, load_config, load_modalities
from architectures.build_architecture import build_architecture
def decode_one_hot_numpy(one_hot_array):
    """
    Giải mã one-hot numpy array có shape (C, D, H, W)
    về lại label shape (D, H, W) với nhãn gốc {0, 1, 2, 3}
    """
    # Tìm nhãn có giá trị lớn nhất trên chiều channel (C)
    label_array = np.argmax(one_hot_array, axis=0)  # shape: (D, H, W)

    # Cộng lại 1 do lúc encode đã trừ đi 1
    label_array = label_array + 1

    # Xác định vùng background (toàn 0 trên chiều channel)
    background_mask = np.sum(one_hot_array, axis=0) == 0
    label_array[background_mask] = 0

    return label_array

def visualize_result(volume, prediction, slice_idx, title):
    # volume: 1, D, H, W
    # prediction: D, H, W
    D, H, W = volume.shape[1:]
    # Initialize a list to store overlayed images
    overlay_all_slices = []

    # Extract the flair modality and the prediction for the given slice
    flair_slice = volume[0, slice_idx, :, :]  # (H, W) - Flair modality
    prediction_slice = prediction[slice_idx, :, :]  # (H, W)
    

    # Convert the flair slice to a 3-channel color image
    flair_slice_color = np.stack((flair_slice,) * 3, axis=-1)  # (H, W, 3)

    # Apply RGB overlays
    flair_slice_color[prediction_slice == 1] = [255, 0, 0]    # RV - Red
    flair_slice_color[prediction_slice == 2] = [0, 255, 0]    # Myo - Green
    flair_slice_color[prediction_slice == 3] = [0, 0, 255]    # LV - Blue

    # Display the slice to check results
    plt.imshow(flair_slice_color)
    plt.title(title)
    plt.axis('off')
    plt.show()

def create_gif_from_volume(volume, prediction, gif_path='output.gif', duration=0.1):
    """
    Tạo ảnh GIF từ volume và prediction. Mỗi frame là một lát cắt với overlay màu.
    - volume: (1, D, H, W)
    - prediction: (D, H, W)
    - gif_path: Đường dẫn lưu file GIF
    - duration: Thời gian hiển thị mỗi frame (giây)
    """
    D = volume.shape[1]
    frames = []

    for slice_idx in range(D):
        flair_slice = volume[0, slice_idx, :, :]
        prediction_slice = prediction[slice_idx, :, :]

        # Chuyển flair slice thành ảnh RGB
        flair_rgb = np.stack((flair_slice,) * 3, axis=-1)

        # Normalize flair để đưa về [0, 255]
        max_val = flair_rgb.max()
        if max_val > 0:
            flair_rgb = np.clip(flair_rgb / max_val, 0, 1) * 255
        else:
            flair_rgb = np.zeros_like(flair_rgb)

        flair_rgb = flair_rgb.astype(np.uint8)

        # Gán màu cho mask prediction
        flair_rgb[prediction_slice == 1] = [255, 0, 0]    # RV - Red
        flair_rgb[prediction_slice == 2] = [0, 255, 0]    # Myo - Green
        flair_rgb[prediction_slice == 3] = [0, 0, 255]    # LV - Blue

        # Chuyển thành ảnh PIL
        frame = Image.fromarray(flair_rgb).convert("RGB")
        frames.append(frame)

    # Lưu GIF
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration * 1000,
            loop=0
        )
        print(f"✅ GIF saved at: {os.path.abspath(gif_path)}")
    else:
        print("⚠️ No frames to save.")


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
    image_pth = "data/acdc_seg/processed/val/patient136_frame01/patient136_frame01_modalities.pt"
    label_pth = "data/acdc_seg/processed/val/patient136_frame01/patient136_frame01_label.pt"
    image = load_modalities(image_pth)
    label = load_modalities(label_pth).astype(int)
    config = load_config("config.yaml")
    prediction = inference(config, image, cpkt="checkpoint/segformer3d-bsm-acdc/model.safetensors")[0]
    prediction = decode_one_hot_numpy(prediction)
    create_gif_from_volume(volume=image, prediction=label, gif_path="assets/acdc/volume_with_label/patient136_frame01.gif")
    create_gif_from_volume(volume=image, prediction=prediction, gif_path="assets/acdc/volume_with_prediction/patient136_frame01.gif")