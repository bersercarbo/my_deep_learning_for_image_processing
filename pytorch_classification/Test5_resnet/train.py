import torch
from torchvision import transforms, datasets
import os.path as osp
import json
import torchvision.models.resnet
from model import resnet50
from model import resnet34
from model import resnext50_32x4d
from tqdm import tqdm

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is {}".format(device))
    data_transfrom = {
        "train":transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_root = '../../data_set/'
    flower_data = osp.join(data_root,"flower_data")
    train_set = datasets.ImageFolder(root=osp.join(flower_data,'train')
                                     , transform=data_transfrom['train'])
    train_num = len(train_set)
    val_set = datasets.ImageFolder(root=osp.join(flower_data,'val'),
                                   transform=data_transfrom['val'])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32,shuffle=True,num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=32, shuffle=False, num_workers=0)
    train_step = len(train_loader)
    class_idx = train_set.class_to_idx
    class_ = dict((value, key) for key, value in class_idx.items())
    jsonClass = json.dumps(class_,indent=4)
    with open("./class_indices.json","w") as f:
        f.write(jsonClass)

    model = resnet34()
    model.load_state_dict(torch.load('./resnet34.pth',map_location='cpu'))

    for m in model.parameters():
        m.requires_grad = False
    in_channels = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=in_channels, out_features=5)
    model = model.to(device)

    params = [m for m in model.parameters() if m.requires_grad]
    optimizer = torch.optim.Adam(params,lr = 0.0001)

    loss_function = torch.nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for index, data in enumerate(train_bar):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "[{}/{}] train loss is {:.3f}".format(epoch+1,epochs,loss)

        model.eval()
        acc_num = 0
        for index, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data
                outputs = model(inputs.to(device))
                predict_y = torch.max(outputs,1)[1]
                acc_num += torch.eq(predict_y,labels.to(device)).sum().cpu().numpy()
        print("train loss is {:.3f}, accuraty is {:.3f}".format(running_loss/train_step,acc_num/len(val_set)))
    print("finished train")
    torch.save(model.state_dict(),'./resnet34dflower.pth')
if __name__ == '__main__':
    main()