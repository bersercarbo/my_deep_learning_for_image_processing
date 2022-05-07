import json
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    imageFile = "../tulip.jpg"
    assert osp.exists(imageFile), "{} image file does not exist."
    im = Image.open(imageFile)
    plt.imshow(im)
    im = data_transform(im)
    im = torch.unsqueeze(im, dim=0).to(device)

    class_file = './class_indices.json'
    with open(class_file,'r') as f:
        class_ = json.load(f)


    net = AlexNet(5)
    net.load_state_dict(torch.load('./bestAlexnet.pth'))
    net.to(device)

    net.eval()
    with torch.no_grad():
        output = net(im)
        predict_y = torch.softmax(torch.squeeze(output), dim=0).cpu()
        cls = torch.argmax(predict_y).numpy()
        p = predict_y[cls].numpy()
        print("predict class is {}, pro is {:.3f}".format(class_[str(cls)], p))

    for i in range(len(predict_y)):
        print("class {} prop is {:.3f}".format(class_[str(i)], predict_y[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()