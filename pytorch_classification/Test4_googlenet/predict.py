import torch
import torch.nn as nn
from model import GoogLeNet
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import json

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is :{}".format(device))
    with open('./class_indices.json','r') as f:
        class_ = json.load(f)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    imageFile = '../tulip.jpg'
    im = Image.open(imageFile)
    plt.imshow(im)

    model = GoogLeNet(num_class=5)
    model.load_state_dict(torch.load('./bestGoogLeNet.pth'))
    model = model.to(device)
    model.eval()

    im = data_transform(im)
    im = torch.unsqueeze(im,dim=0).to(device)
    with torch.no_grad():
        output = model(im)
        output = torch.squeeze(output)
        predict_y = torch.softmax(output,dim=0).cpu()
        cla = torch.argmax(predict_y).numpy()
    print("class is {}, pro is {:.3f}".format(class_[str(cla)], predict_y[cla]))

    for i in range(len(predict_y)):
        print("class {} pro is {:.3f}".format(class_[str(i)], predict_y[i]))
    plt.show()

if __name__ == '__main__':
    main()