import torch
from PIL import Image
import numpy as np
import cv2
import argparse

from model import TinyFaceBoxNet, IMG_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_bbox(model, path):
    model.eval()
    img = Image.open(path).convert('RGB')
    w0, h0 = img.size
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.array(img_resized)).permute(2,0,1).float()/255.0
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logit_p, box = model(x)
        p = torch.sigmoid(logit_p)[0,0].item()
        box = box[0].cpu().numpy()

    sx, sy = w0/IMG_SIZE, h0/IMG_SIZE
    x1, y1, x2, y2 = box
    x1*=sx; x2*=sx; y1*=sy; y2*=sy
    return p, (int(x1), int(y1), int(x2), int(y2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to image")
    args = parser.parse_args()

    model = TinyFaceBoxNet().to(device)
    model.load_state_dict(torch.load("models/facebox_cnn.pt", map_location=device))

    p, (x1,y1,x2,y2) = predict_bbox(model, args.image)
    print(f"Probability face: {p:.3f}, BBox: {(x1,y1,x2,y2)}")

    img = cv2.imread(args.image)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
