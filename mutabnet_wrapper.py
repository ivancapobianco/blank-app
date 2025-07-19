import torch
import cv2
import numpy as np
import pandas as pd
from mtb.dataset import TableVirtuoso  # adjust import per repo structure
from mtb.models import TableMaster  # or the actual MuTAbNet model class

class MuTAbNet:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name="mutabnet-lab", device="cpu"):
        model = TableMaster.from_pretrained(model_name)  # adjust naming
        return cls(model, device)

    def predict(self, image_path: str) -> list[pd.DataFrame]:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        # forward pass & decode (adjust per API)
        output = self.model(tensor)
        tables = TableVirtuoso.decode(output, img.shape[:2])

        dfs = []
        for tbl in tables:
            rows = tbl.cells  # you’ll need to adapt this shape
            df = pd.DataFrame(rows)
            dfs.append(df)

        return dfs