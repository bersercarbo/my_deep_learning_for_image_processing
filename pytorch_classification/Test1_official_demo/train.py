import torch
import torch.optim as optimizer
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import LeNet

def main():
    Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    Train_Set = datasets.CIFAR10(root='./data', download=False,
                                 transform=Transform,train=True)
    train_loader = torch.utils.data.DataLoader(dataset=Train_Set, batch_size=32,
                                               shuffle=True,num_workers=0)

    Val_Set = datasets.CIFAR10(root='./data', download=False,
                               transform=Transform,train=False)
    val_loader = torch.utils.data.DataLoader(dataset=Val_Set, batch_size=10000,
                                             shuffle=False,num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
    net = LeNet()
    opt = optimizer.Adam(net.parameters(),lr=0.001)
    LossFunction = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        running_loss = 0.0

        for index, data in enumerate(train_loader,start=0):
            inputs, labels = data
            opt.zero_grad()
            outputs = net(inputs)
            loss = LossFunction(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if  index % 500 == 499 :
                with torch.no_grad():
                    outputs = net(val_image)
                    predict_y = torch.max(outputs,dim=1)[1]
                    accuracy = torch.eq(predict_y,val_label).sum().item() / val_label.size(0)
                    print("[{} / {}] train lossing is {}, test accuracy is {}"
                          .format(epoch+1, index+1, running_loss / 500, accuracy))
    print("train finished")
    save_path = "./lenet.pth"
    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()