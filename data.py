# system
import glob
import numpy as np

# torch
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, root, cfg):
        self.modality1_files = np.array(sorted(glob.glob("%s/*_modality1.npy" % root)))
        self.modality2_files = np.array(sorted(glob.glob("%s/*_modality2.npy" % root)))
        self.cfg = cfg

    def __getitem__(self, index):
        modality1_path = self.modality1_files[index % len(self.modality1_files)]
        modality2_path = self.modality2_files[index % len(self.modality2_files)]
        modality1 = torch.from_numpy(np.load(modality1_path)).to(torch.float32)
        modality2 = torch.from_numpy(np.load(modality2_path)).to(torch.float32)
        return modality1, modality2
    def __len__(self):
        return len(self.modality1_files)


class ValDataset(Dataset):
    def __init__(self, root, cfg):
        self.modality1_files = np.array(sorted(glob.glob("%s/*modality1.npy" % root)))
        self.modality2_files = np.array(sorted(glob.glob("%s/*modality2.npy" % root)))
        self.modality1_seg_files = np.array(sorted(glob.glob("%s/*modality1_seg.npy" % root)))
        self.modality2_seg_files = np.array(sorted(glob.glob("%s/*modality2_seg.npy" % root)))
        self.cfg = cfg

    def __getitem__(self, index):
        modality1_path = self.modality1_files[index % len(self.modality1_files)]
        modality2_path = self.modality2_files[index % len(self.modality2_files)]
        modality1_seg_path = self.modality1_seg_files[index % len(self.modality1_seg_files)]
        modality2_seg_path = self.modality2_seg_files[index % len(self.modality2_seg_files)]
        modality1 = torch.from_numpy(np.load(modality1_path)).to(
            torch.float32)
        modality2 = torch.from_numpy(np.load(modality2_path)).to(
            torch.float32)
        modality1_seg = torch.from_numpy(np.load(modality1_seg_path)).to(
            torch.float32)
        modality2_seg = torch.from_numpy(np.load(modality2_seg_path)).to(
            torch.float32)
        return modality1, modality2, modality1_seg, modality2_seg

    def __len__(self):
        return len(self.modality1_files)
