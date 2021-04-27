import time

import pandas as pd
import timm
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms


def predict_image(path):
    print("Prediction in progress")
    image = Image.open(path)

    transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_tensor = transformation(image).float()
    img_tensor = img_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    start = time.time()
    input = Variable(img_tensor)
    output = model(input)
    index = output.cpu().data.numpy().argmax()
    end = time.time()
    print("[INFO] Prediction took {:.5f} seconds".format(
        end - start))
    # prob = F.softmax(output, dim=1)
    return index


if __name__ == "__main__":
    imgpath = "data/2.jpg"
    model = timm.create_model('efficientnet_v2s', num_classes=7)
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load("models/model_best.pth-d043d179.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    index = predict_image(imgpath)
    df = pd.read_csv("file_index.csv")
    pred = df.iloc[index][1]
    print("Predicted Class ", pred)
