import os 
import torch
from torch.utils.data import Dataset 
from utils.utils import (
    load_modalities
)
def one_hot_encode_label(label_tensor, num_classes=3):
    label_tensor = label_tensor.clone()

    # Mask background (label == 0)
    mask = label_tensor == 0

    # Shift labels: 1->0, 2->1, 3->2 (background remains 0 for now)
    label_tensor = label_tensor - 1

    # Temporarily set background (now -1) to 0 just to pass into one_hot
    label_tensor[mask] = 0

    # One-hot encode
    one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=num_classes)

    # Zero out the background again
    one_hot[mask] = 0

    return one_hot.float()

class ACDCDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        root_dir: path to folder where contain casenames
        transform: composition of the pytorch transforms 
        """
        super().__init__()
        self.root_dir = root_dir
        self.case_names = next(os.walk(self.root_dir), (None, None, []))[1]
        self.transform = transform

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx] 
        
        volume_fp = os.path.join(self.root_dir, case_name, f"{case_name}_modalities.pt")
        label_fp = os.path.join(self.root_dir, case_name, f"{case_name}_label.pt")
        volume = load_modalities(volume_fp)
        label = load_modalities(label_fp)
        volume_tensor = torch.from_numpy(volume).float()
        label_tensor = torch.from_numpy(label).long()
        label_tensor = label_tensor.squeeze(0)
        # One-hot encode label
        label_one_hot = one_hot_encode_label(label_tensor).permute(3, 0, 1, 2)
        data = {"image": volume_tensor, "label": label_one_hot}
        if self.transform:
            data = self.transform(data)
        return data
