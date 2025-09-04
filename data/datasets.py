import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        
        # Handle X - convert to float, handling strings if present
        X_raw = data["X"]
        if np.issubdtype(X_raw.dtype, np.str_) or X_raw.dtype == object:
            # If X contains strings, convert them to float
            try:
                self.X = np.array(X_raw, dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Cannot convert X data to float: {e}")
        else:
            self.X = np.array(X_raw, dtype=np.float32)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
        # Handle y - convert to integer (handles strings in y)
        y_raw = data["y"]
        if np.issubdtype(y_raw.dtype, np.str_) or y_raw.dtype == object:
            classes = np.unique(y_raw)
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
            y_numeric = np.array([self.class_to_idx[label] for label in y_raw], dtype=np.int64)
        else:
            y_numeric = np.array(y_raw, dtype=np.int64)
            self.class_to_idx = None
            
        self.y = torch.tensor(y_numeric, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_class_mapping(self):
        return self.class_to_idx

def get_dataloader(npz_path, batch_size=32, shuffle=True, num_workers=0):
    dataset = PoseDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)