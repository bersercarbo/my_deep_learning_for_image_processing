import os
import os.path as osp
import sys
import json

import torch
from torchvision import datasets, transforms

from model import vgg
from tqdm import tqdm

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device is {}".format(device))

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val":  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    data_root = osp.abspath(osp.join(os.getcwd(), '../../data_set'))
    flowers_data = osp.join(data_root, 'flower_data')
    assert osp.exists(flowers_data), "flower path {} does not exist.".format(flowers_data)
    assert osp.exists(osp.join(flowers_data, 'train')), "flower train does not exist."
    assert osp.exists(osp.join(flowers_data, 'val')), "flower val does not exist."

    train_set = datasets.ImageFolder(root=osp.join(flowers_data, "train"), transform=data_transforms["train"])
    val_set = datasets.ImageFolder(root=osp.join(flowers_data, 'val'), transform=data_transforms['val'])
    train_num = len(train_set)
    val_num = len(val_set)
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    train_steps = len(train_loader)
    print("train image: {}, val image: {}".format(train_num,val_num))

    class_idx = train_set.class_to_idx
    class_dict = dict((value, key) for key, value in class_idx.items())
    class_json = json.dumps(class_dict, indent=4)
    with open("class_indices.json", 'w') as f:
        f.write(class_json)

    model_name = "vgg16"

    model = vgg(model_name, init_weights=True, num_class=5)
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 10
    best_accurate = 0.0
    for epoch in range(epochs):
        model.train()
        runing_loss = 0.0
        train_bar=tqdm(train_loader,file=sys.stdout)
        for index, data in enumerate(train_bar):
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            runing_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_bar.desc = "[{}/{}]Train Process:loss is {:.3f}".format(epoch+1, epochs, loss.item())

        model.eval()
        #val_bar=tqdm(val_loader, file=sys.stdout)
        with torch.no_grad():
            acc = 0
            for index, data in enumerate(val_loader):
                inputs, labels = data
                outputs=model(inputs.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc+=torch.eq(predict_y,labels.to(device)).sum().item()
            cur_accurate = acc / val_num
            if cur_accurate > best_accurate:
                torch.save(model.state_dict(), './best{}_{}.pth'.format(model_name,epoch+1))
            print("epoch{} train loss is {:.3f}, accurate is {:.3f}".format(epoch+1, loss, cur_accurate))

        torch.save(model.state_dict(), './final{}.pth'.format(model_name))
    print("Train Finished!")

if __name__ == '__main__':
    main()