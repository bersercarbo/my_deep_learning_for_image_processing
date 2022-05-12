import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from model import vgg
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is :{}".format(device))
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    im = Image.open('../tulip.jpg')
    plt.imshow(im)
    im = data_transform(im)
    im = torch.unsqueeze(im,dim=0).to(device)

    with open('./class_indices.json') as f:
        classIndices = json.load(f)

    model = vgg("vgg16", init_weights=True, num_class=5)
    model.load_state_dict(torch.load('./finalvgg16.pth'))
    model.to(device)

    model.eval()
    with torch.no_grad():
        output = model(im)
        output = torch.squeeze(output)
        predict_y = torch.softmax(output,dim=0).cpu()
        cla = torch.argmax(predict_y).numpy()
        p = predict_y[cla].numpy()
        print("predict class is {}, pro if {}".format(classIndices[str(cla)], p))

    for i in range(len(predict_y)):
        print("class {} : p {}".format(classIndices[str(i)], predict_y[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()