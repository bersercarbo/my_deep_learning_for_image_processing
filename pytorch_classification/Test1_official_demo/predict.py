import torch
from PIL import Image
import torchvision.transforms as transforms
from model import LeNet
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    transforms.Resize((32,32))
])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('./lenet.pth'))

test_image = Image.open('./dog.jpg')
im = transform(test_image)
im = torch.unsqueeze(im, dim=0)

with torch.no_grad():
    predict = net(im)
    print(predict)
    print(torch.max(predict,dim=1))
    print(classes[torch.max(predict,dim=1)[1]])