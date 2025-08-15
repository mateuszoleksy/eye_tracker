import os, csv, random
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

IMG_SIZE = 256

class CelebAFaceBox(Dataset):
    def __init__(self, img_dir, csv_path, train=True):
        self.img_dir = img_dir
        self.items = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                x1 = float(r['x_1']); y1 = float(r['y_1'])
                w  = float(r['width']);  h  = float(r['height'])
                x2 = x1 + w; y2 = y1 + h
                self.items.append((r['image_id'], x1, y1, x2, y2))
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, x1, y1, x2, y2 = self.items[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert('RGB')
        w0, h0 = img.size

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        sx, sy = IMG_SIZE / w0, IMG_SIZE / h0
        x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy

        if self.train:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x1_, x2_ = IMG_SIZE - x2, IMG_SIZE - x1
                x1, x2 = x1_, x2_

        img_t = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
        target = torch.tensor([1.0, x1, y1, x2, y2], dtype=torch.float32)
        return img_t, target
