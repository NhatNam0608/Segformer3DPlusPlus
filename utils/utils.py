import os
import torch
import imageio
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from monai.data import MetaTensor
from sklearn.preprocessing import MinMaxScaler 
from monai.transforms import (
    Orientation, 
    EnsureType,
)
import yaml
import random
import math
from PIL import Image
def load_nifti(fp:str) -> list:
    """
    Load a NIfTI file and return its data array and affine matrix.
    
    Parameters:
        fp (str): Path to the NIfTI file (.nii or .nii.gz).
    
    Returns:
        tuple: (nifti_scan, affine) if successful, otherwise None.
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File {fp} does not exist.")
    
    try:
        nifti_data = nib.load(fp)
        nifti_scan = nifti_data.get_fdata() 
        affine = nifti_data.affine 
        return nifti_scan, affine
    except Exception as e:
        raise RuntimeError(f"Error loading NIfTI file {fp}: {e}")
def load_modalities(file_path: str):
    """
    Load the modalities tensor from a .pt file.

    Args:
        file_path (str): Đường dẫn tới file .pt

    Returns:
        torch.Tensor hoặc None nếu có lỗi.
    """
    try:
        modalities = torch.load(file_path, map_location=torch.device("cpu"), weights_only=False)
        return modalities
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
def detach_meta(x: MetaTensor) -> np.ndarray:
        if not isinstance(x, MetaTensor):
            raise TypeError("Input to `detach_meta` must be a MetaTensor.")
        return EnsureType(data_type="numpy", track_meta=False)(x)
def orient(x: MetaTensor) -> MetaTensor:
        if not isinstance(x, MetaTensor):
            raise TypeError("Input to `orient` must be a MetaTensor.")
        return Orientation(axcodes="RAS")(x)   
def normalize(x: np.ndarray) -> np.ndarray:
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
            return normalized_1D_array.reshape(x.shape)
        except ValueError as e:
            raise ValueError(f"Error in normalization: {e}")
def generate_gif(x: np.ndarray, fp: str):
    imageio.mimsave(fp, (x * 255).astype(np.uint8), duration=0.1)
     
def seed_everything(seed) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """loads the yaml config file

    Args:
        config_path (str): _description_

    Returns:
        Dict: _description_
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
def visualize_result(volume, prediction, slice_idx, title):
    # volume: 3, D, H, W
    # prediction: 3, D, H, W
    D, H, W = volume.shape[1:]

    # Initialize a list to store overlayed images
    overlay_all_slices = []

    # Extract the flair modality and the prediction for the given slice
    flair_slice = volume[0, slice_idx, :, :]  # (H, W) - Flair modality
    prediction_slice = prediction[:, slice_idx, :, :]  # (3, H, W)

    # Extract WT, TC, ET masks
    wt_mask = prediction_slice[1, :, :]  # Channel 2: Whole Tumor (WT)
    tc_mask = prediction_slice[0, :, :]  # Channel 1: Tumor Core (TC)
    et_mask = prediction_slice[2, :, :]  # Channel 3: Enhancing Tumor (ET)

    # Overlay masks in priority order: ET > TC > WT
    final_mask = np.zeros_like(wt_mask)

    final_mask[et_mask > 0] = 3  # ET
    final_mask[(tc_mask > 0) & (final_mask == 0)] = 2  # TC
    final_mask[(wt_mask > 0) & (final_mask == 0)] = 1  # WT

    # Convert the flair slice to a 3-channel color image
    flair_slice_color = np.stack((flair_slice,) * 3, axis=-1)  # (H, W, 3)

    # Apply RGB overlays
    flair_slice_color[final_mask == 1] = [255, 0, 0]    # WT - Red
    flair_slice_color[final_mask == 2] = [0, 255, 0]    # TC - Green
    flair_slice_color[final_mask == 3] = [0, 0, 255]    # ET - Blue

    # Display the slice to check results
    plt.imshow(flair_slice_color)
    plt.title(title)
    plt.axis('off')
    plt.show()

def create_gif_from_volume(volume, prediction, modal=0, gif_path='output.gif', duration=0.1):
    """
    Create a GIF from all slices along the D dimension of the volume and prediction.
    - volume: tensor of shape (3, D, H, W)
    - prediction: tensor of shape (3, D, H, W)
    """
    _, D, H, W = volume.shape
    frames = []

    for slice_idx in range(D):
        flair_slice = volume[modal, slice_idx, :, :]
        prediction_slice = prediction[:, slice_idx, :, :]

        wt_mask = prediction_slice[1, :, :]
        tc_mask = prediction_slice[0, :, :]
        et_mask = prediction_slice[2, :, :]

        final_mask = np.zeros_like(wt_mask)
        final_mask[et_mask > 0] = 3
        final_mask[(tc_mask > 0) & (final_mask == 0)] = 2
        final_mask[(wt_mask > 0) & (final_mask == 0)] = 1

        flair_rgb = np.stack((flair_slice,) * 3, axis=-1)

        # Normalize and scale to [0, 255]
        max_val = flair_rgb.max()
        if max_val > 0:
            flair_rgb = np.clip(flair_rgb / max_val, 0, 1) * 255
        else:
            flair_rgb = np.zeros_like(flair_rgb)  # fallback to black image

        flair_rgb = flair_rgb.astype(np.uint8)

        # Apply color masks
        flair_rgb[final_mask == 1] = [255, 0, 0]  # WT - Red
        flair_rgb[final_mask == 2] = [0, 255, 0]  # TC - Green
        flair_rgb[final_mask == 3] = [0, 0, 255]  # ET - Blue

        # Convert to PIL image
        img_pil = Image.fromarray(flair_rgb)
        img_pil = img_pil.convert("RGB")

        # Optional: add text like "Slice: idx" using ImageDraw, ImageFont

        frames.append(img_pil)

    # Save as animated GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration * 1000,
        loop=0
    )
    print(f"✅ GIF saved at: {gif_path}")
def cube_root(n):
    return round(math.pow(n, (1 / 3)))
def create_tensor_from_volume(volume, prediction, modal=0):
    """
    Tạo tensor hình ảnh RGB từ các lát cắt trong volume và prediction.
    - volume: tensor (3, D, H, W)
    - prediction: tensor (3, D, H, W)
    
    Trả về: tensor (D, H, W, 3) dtype=torch.uint8
    """
    _, D, H, W = volume.shape
    output = []
    for slice_idx in range(D):
        flair_slice = volume[modal, slice_idx, :, :]
        prediction_slice = prediction[:, slice_idx, :, :]

        wt_mask = prediction_slice[1, :, :]
        tc_mask = prediction_slice[0, :, :]
        et_mask = prediction_slice[2, :, :]

        final_mask = np.zeros_like(wt_mask)
        final_mask[et_mask > 0] = 3
        final_mask[(tc_mask > 0) & (final_mask == 0)] = 2
        final_mask[(wt_mask > 0) & (final_mask == 0)] = 1

        flair_rgb = np.stack((flair_slice,) * 3, axis=-1)

        # Normalize to [0, 255]
        max_val = flair_rgb.max()
        if max_val > 0:
            flair_rgb = np.clip(flair_rgb / max_val, 0, 1) * 255
        else:
            flair_rgb = np.zeros_like(flair_rgb)

        flair_rgb = flair_rgb.astype(np.uint8)

        # Apply color masks
        flair_rgb[final_mask == 1] = [255, 0, 0]   # WT - Red
        flair_rgb[final_mask == 2] = [0, 255, 0]   # TC - Green
        flair_rgb[final_mask == 3] = [0, 0, 255]   # ET - Blue

        output.append(flair_rgb)

    # Convert list of (H, W, 3) arrays to torch tensor (D, H, W, 3)
    rgb_tensor = torch.from_numpy(np.stack(output)).byte()
    return rgb_tensor  # Shape: (D, H, W, 3)
if __name__ == "__main__":
    case_names = next(os.walk("data/acdc_seg/processed"), (None, None, []))[1]
    for case_name in case_names:
        print(f"Processing case: {case_name}")
        try:
            data_fp = f"data/acdc_seg/processed/{case_name}/{case_name}_modalities.pt"
            data = load_modalities(data_fp)
            if data is not None:
                print(f"Loaded modalities for {case_name} successfully.")
                # Example of generating a GIF for the first modality
                generate_gif(data[0], fp=f"assets/acdc_seg/preprocess/{case_name}_preprocessed.gif")
            else:
                print(f"Failed to load modalities for {case_name}.")
        except Exception as e:
            print(f"Error processing {case_name}: {e}")
    # data = load_modalities("data/acdc_seg/processed/patient001_frame01/patient001_frame01_modalities.pt")
    # generate_gif(data[0], fp="assets/acdc_seg/preprocess/patient001_frame01_preprocessed.gif")
    # data = load_modalities("data/acdc_seg/processed/patient038_frame11/patient038_frame11_modalities.pt")
    # label = load_modalities("data/acdc_seg/processed/patient038_frame11/patient038_frame11_label.pt")
    # create_gif_from_volume(data, label)
    # data, _ = load_nifti("data/acdc_seg/raw/patient022_frame01/patient022_frame01.nii.gz")
    # print(data.shape)
    # label, _ = load_nifti("data/acdc_seg/raw/patient057_frame01/patient057_frame01_gt.nii.gz")
    # data_normalized = normalize(data)
    # label_normalized = normalize(label)
    # data_transformed = np.transpose(data_normalized, (2, 0, 1))  # Convert to (D, H, W)
    # label_transformed = np.transpose(label_normalized, (2, 0, 1))  # Convert to (D, H, W)
    # print(f"Data shape: {data_transformed.shape}, Label shape: {label_transformed.shape}")
    # generate_gif(data_transformed, fp="output_data.gif")
    # generate_gif(label_transformed, fp="output_label.gif")
    # import os
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # Load dữ liệu
    # case_names = next(os.walk("data/acdc_seg/raw"), (None, None, []))[1]

    # max_H, max_W = 0, 0
    # for case_name in case_names:
    #     label_fp = os.path.join("data/acdc_seg/raw", case_name, f"{case_name}_gt.nii.gz")
    #     label, _ = load_nifti(label_fp)
    #     h, w = label.shape[:2]
    #     max_H = max(max_H, h)
    #     max_W = max(max_W, w)

    # # Bước 2: Khởi tạo heatmap
    # heatmap = np.zeros((max_H, max_W), dtype=np.int32)

    # # Bước 3: Duyệt và cộng dồn mask
    # for case_name in case_names:
    #     label_fp = os.path.join("data/acdc_seg/raw", case_name, f"{case_name}_gt.nii.gz")
    #     label, _ = load_nifti(label_fp)
    #     label = label.astype(np.uint8)

    #     # Tạo mask nhãn khác 0
    #     if label.ndim == 3:
    #         mask = (label > 0).any(axis=-1)  # gộp theo chiều D
    #     else:
    #         mask = (label > 0)

    #     h, w = mask.shape
    #     heatmap[:h, :w] += mask.astype(np.int32)

    # # Bước 4: Vẽ heatmap
    # plt.figure(figsize=(10, 8))
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Số lần xuất hiện pixel có nhãn 1, 2, 3')
    # plt.title('Phân phối không gian của các nhãn khác 0 trong tất cả ảnh')
    # plt.xlabel('Trục W (chiều ngang)')
    # plt.ylabel('Trục H (chiều dọc)')
    # plt.tight_layout()
    # plt.show()
    # max_H = 0
    # max_W = 0
    # max_H_case = ""
    # max_W_case = ""

    # # Duyệt từng case
    # for case_name in case_names:
    #     label_fp = os.path.join("data/acdc_seg/raw", case_name, f"{case_name}_gt.nii.gz")
    #     label, _ = load_nifti(label_fp)
    #     h, w = label.shape[:2]

    #     if h > max_H:
    #         max_H = h
    #         max_H_case = case_name

    #     if w > max_W:
    #         max_W = w
    #         max_W_case = case_name

    # # In kết quả
    # print(f"🟩 Case có H lớn nhất: {max_H_case} (H = {max_H})")
    # print(f"🟦 Case có W lớn nhất: {max_W_case} (W = {max_W})")
    # min_H = float('inf')
    # min_W = float('inf')
    # min_D = float('inf')
    # min_H_case = ""
    # min_W_case = ""
    # min_D_case = ""

    # # Duyệt từng case
    # for case_name in case_names:
    #     label_fp = os.path.join("data/acdc_seg/raw", case_name, f"{case_name}_gt.nii.gz")
    #     label, _ = load_nifti(label_fp)
    #     h, w, d = label.shape  # Lấy H, W, D

    #     if h < min_H:
    #         min_H = h
    #         min_H_case = case_name

    #     if w < min_W:
    #         min_W = w
    #         min_W_case = case_name

    #     if d < min_D:
    #         min_D = d
    #         min_D_case = case_name

    # # In kết quả
    # print(f"🟥 Case có H nhỏ nhất: {min_H_case} (H = {min_H})")
    # print(f"🟨 Case có W nhỏ nhất: {min_W_case} (W = {min_W})")
    # print(f"🟦 Case có D nhỏ nhất: {min_D_case} (D = {min_D})")

