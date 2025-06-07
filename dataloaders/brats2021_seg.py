import os 
import torch
from torch.utils.data import Dataset 

from utils.utils import (
    load_modalities
)


class Brats2021SegDataset(Dataset):
	def __init__(self, root_dir: str, transform = None):
		"""
		root_dir: path to folder where contain casenames
		transform: composition of the pytorch transforms 
		"""
		super().__init__()
		self.root_dir = root_dir
		self.case_names = next(os.walk(self.root_dir), (None, None, []))[1]
		self.transform = transform

	def __len__(self):
		return self.case_names.__len__()

	def __getitem__(self, idx):
		case_name = self.case_names[idx] 
		
		volume_fp = os.path.join(self.root_dir, case_name, f"{case_name}_modalities.pt")
		label_fp = os.path.join(self.root_dir, case_name, f"{case_name}_label.pt")
		volume = load_modalities(volume_fp)
		label = load_modalities(label_fp)
		data = {"image": torch.from_numpy(volume).float(), "label": torch.from_numpy(label).float()}
		if self.transform:
			data = self.transform(data)
		return data