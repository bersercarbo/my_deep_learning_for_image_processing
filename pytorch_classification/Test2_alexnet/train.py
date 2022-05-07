import os
import os.path as osp
import json
import sys

from torchvision import datasets, transforms
import torch
import torch.nn as nn
from model import AlexNet
from tqdm import tqdm


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {

        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),

        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_root = osp.abspath(osp.join(os.getcwd(), '../../data_set'))
    flower_data = osp.join(data_root,"flower_data")
    assert osp.exists(flower_data), "{} path does not exist".format(flower_data)
    assert osp.exists(osp.join(flower_data,'train')), "{}, train data does not exist".format(flower_data)
    assert osp.exists(osp.join(flower_data,'val')), "{}, val data does not exist".format(flower_data)
    train_set = datasets.ImageFolder(root=osp.join(flower_data,'train'),transform=data_transform['train'])
    train_num = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32,
                                               num_workers = 0, shuffle = True)
    flower_list = train_set.class_to_idx
    class_dict = dict((val, key) for key, val in flower_list.items())
    json_class = json.dumps(class_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_class)

    val_set = datasets.ImageFolder(root=osp.join(flower_data, 'val'),transform=data_transform['val'])
    val_num = len(val_set)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 4,
                                             num_workers = 0, shuffle = True)

    print("using {} images for train, {} images for val".format(train_num, val_num))

    train_steps = len(train_loader)
    epochs = 10
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_model_file = './bestAlexnet.pth'
    final_model_file = './finalAlexnet.pth'
    net = AlexNet(5,True)
    net.to(device)
    optimizer = torch.optim.Adam(params=net.parameters(),lr=0.0002)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader,file=sys.stdout)
        for index, data in enumerate(train_bar):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch [{}/{}] loss is {:.3f}.".format(epoch+1,epochs,loss)

        net.eval()
        accuray = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for index, data in enumerate(val_bar):
                inputs, labels = data
                outputs = net(inputs.to(device))
                predicts = torch.max(outputs, dim=1)[1]
                accuray += torch.eq(predicts,labels.to(device)).sum().item()
            val_accurate =  accuray / val_num
            if val_accurate > best_acc:
                best_acc = accuray
                torch.save(net.state_dict(),best_model_file)
            print('epoch {} train lossing is {:.3f}, accurate is {:.3f}'
                  .format(epoch+1, running_loss/train_steps,val_accurate))

        torch.save(net.state_dict(),final_model_file)
        print("")
    print("train finished!")

if __name__ == '__main__':
    main()
