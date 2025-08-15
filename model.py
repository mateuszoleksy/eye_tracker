import torch
import torch.nn as nn

IMG_SIZE = 256

class TinyFaceBoxNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3,2,1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,2,1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(256, 5)

    def forward(self, x):
        b = x.size(0)
        f = self.feat(x).view(b, -1)
        out = self.head(f)
        logit_p = out[:, :1]
        box = out[:, 1:]
        box = torch.clamp(box, 0.0, float(IMG_SIZE))
        return logit_p, box

def iou_xyxy(boxA, boxB, eps=1e-6):
    xA = torch.max(boxA[:,0], boxB[:,0])
    yA = torch.max(boxA[:,1], boxB[:,1])
    xB = torch.min(boxA[:,2], boxB[:,2])
    yB = torch.min(boxA[:,3], boxB[:,3])
    inter = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    areaA = torch.clamp(boxA[:,2]-boxA[:,0], min=0) * torch.clamp(boxA[:,3]-boxA[:,1], min=0)
    areaB = torch.clamp(boxB[:,2]-boxB[:,0], min=0) * torch.clamp(boxB[:,3]-boxB[:,1], min=0)
    union = areaA + areaB - inter + eps
    return inter / union
