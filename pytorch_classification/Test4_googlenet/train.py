import torch
import torch.nn as nn
from torchvision import transforms,datasets
import json
import os.path as osp
from model import GoogLeNet
from tqdm import tqdm

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is {}".format(device))
    data_transforms = {
        "train":transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]),
        "val"  :transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }

    data_root = "../../data_set/flower_data"
    train_data = datasets.ImageFolder(root=osp.join(data_root,"train"),
                                      transform=data_transforms["train"])
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = 32,shuffle = True,num_workers = 0)
    val_data = datasets.ImageFolder(root=osp.join(data_root,"val"),
                                    transform=data_transforms["val"])
    val_loader = torch.utils.data.DataLoader(dataset=val_data,batch_size=32,num_workers = 0)

    train_step = len(train_loader)
    val_step = len(val_loader)
    train_num = len(train_data)
    val_num = len(val_data)
    print("train num is {}, val num is {}.".format(train_num, val_num))
    classIdx = train_data.class_to_idx
    class_ = dict((value,key) for key, value in classIdx.items())
    jsonclass = json.dumps(class_,indent=4)
    with open("./class_indices.json","w") as f:
        f.write(jsonclass)

    model = GoogLeNet(num_class=5,init_weights=True)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    epochs = 30
    best_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        tran_bar = tqdm(train_loader)
        model.train()
        for index, data in enumerate(tran_bar):
            optimizer.zero_grad()
            inputs, labels = data
            outputs, aux1, aux2 = model(inputs.to(device))
            labels = labels.to(device)
            loss1 = loss_function(outputs,labels)
            loss2 = loss_function(aux1, labels)
            loss3 = loss_function(aux2, labels)
            losstatol = loss1+0.3*loss2+0.3*loss3
            running_loss += losstatol.item()
            losstatol.backward()
            optimizer.step()
            tran_bar.desc = "[{}/{}] train loss is {:.3f}".format(epoch+1,epochs,losstatol.item())
        model.eval()
        accurate = 0.0
        for index, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data
                outputs = model(inputs.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                accurate += torch.eq(predict_y,labels.to(device)).sum().item()
        acc = accurate / val_num
        print("epoch{} train loss is {:.3f}, accurate is {:.3f}".
              format(epoch+1, running_loss/train_step, acc))
        if  acc > best_acc :
            best_acc = acc
            torch.save(model.state_dict(),'./bestGoogLeNet.pth')
        torch.save(model.state_dict(),'./finalGoogLeNet.pth')
    print("train finished")

if __name__ == '__main__':
    main()