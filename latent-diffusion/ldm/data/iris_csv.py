import cv2, numpy as np, torch
from torch.utils.data import Dataset
import pandas as pd

def imread_gray3(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert g is not None, path
    g = g.astype(np.float32) / 255.0            # [H,W] in [0,1]
    img = np.stack([g, g, g], axis=-1)           # -> [3,H,W] (CHW)
    img = img * 2.0 - 1.0                       # -> [-1,1]
    return torch.from_numpy(img)

def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE); assert m is not None, path
    m = (m > 127).astype(np.float32)
    m = m[None, ...].copy()                   # [1,H,W]  (随你保留CHW，AE阶段用不到)
    return torch.from_numpy(m)

class IrisCSV(Dataset):
    def __init__(self, csv_path, split="train"):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img  = imread_gray3(r["img_path"])    # [H,W,3]  (HWC)
        mask = read_mask(r["mask_path"])      # [1,H,W]
        y    = int(r["class_id"])
        human = str(r["class_name"]) if "class_name" in self.df.columns else f"class_{y}"
        return {"image": img, "mask": mask, "class_label": y, "human_label": human}