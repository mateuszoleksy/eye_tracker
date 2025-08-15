import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn

from dataset import CelebAFaceBox
from model import TinyFaceBoxNet, iou_xyxy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm import tqdm

def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0
    total_iou  = 0.0
    total_n    = 0

    loop = tqdm(loader, desc="Train" if train else "Val")
    for imgs, tgt in loop:
        imgs = imgs.to(device)
        tgt  = tgt.to(device)
        t_p  = tgt[:, :1]
        t_box= tgt[:, 1:]

        if train:
            opt.zero_grad()
        logit_p, box = model(imgs)

        loss_p = bce(logit_p, t_p)
        loss_box = l1(box, t_box)
        loss = loss_p + 2.0*loss_box

        if train:
            loss.backward()
            opt.step()

        with torch.no_grad():
            iou = iou_xyxy(box, t_box).mean()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_iou  += iou.item() * bs
        total_n    += bs

        avg_loss = total_loss / total_n
        avg_iou = total_iou / total_n
        loop.set_postfix(loss=f"{avg_loss:.4f}", iou=f"{avg_iou:.3f}")

    return total_loss/total_n, total_iou/total_n


if __name__ == '__main__':
    full = CelebAFaceBox("data/img", "data/bbox.csv", train=True)
    n = len(full)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_ds.dataset.train = False

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyFaceBoxNet().to(device)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    l1  = nn.SmoothL1Loss()

    EPOCHS = 20
    best_val = 1e9
    for e in range(1, EPOCHS+1):
        tr_loss, tr_iou = run_epoch(train_loader, train=True)
        va_loss, va_iou = run_epoch(val_loader, train=False)
        print(f"Ep {e:02d} | train {tr_loss:.4f} IoU {tr_iou:.3f} | val {va_loss:.4f} IoU {va_iou:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "models/facebox_cnn.pt")
