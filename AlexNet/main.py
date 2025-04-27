import torch
import torch.nn as nn
import CNN
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224,antialias=True),
    transforms.Normalize(0.1307, 0.3081)
])

train_data=datasets.MNIST(
    root='C:\\Users\\LENOVO\\Desktop\\code\\dataset\\MNIST',
    train=True,
    download=True,
    transform=transform
)

test_data=datasets.MNIST(
    root='C:\\Users\\LENOVO\\Desktop\\code\\dataset\\MNIST',
    train=False,
    download=True,
    transform=transform
)


train_loader=DataLoader(dataset=train_data,shuffle=True,batch_size=128)
test_loader=DataLoader(dataset=test_data,shuffle=False,batch_size=128)

model=CNN.CNN().to('cuda:0')

epoch=10
learning_rate=0.1
losses=[]
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epochs in range(epoch):
    for(x,y)in train_loader:
        x,y=x.to('cuda:0'),y.to('cuda:0')
        Pred=model(x)
        loss=loss_fn(Pred,y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

fig=plt.figure()
plt.plot(range(len(losses)),losses)
plt.show()

correct=0
total=0
with torch.no_grad():
    for(x,y)in test_loader:
        x,y=x.to('cuda:0'),y.to('cuda:0')
        Pred=model(x)
        a,predicted=torch.max(Pred.data,dim=1)
        correct+=torch.sum((predicted==y))
        total+=y.size(0)
print(f'测试集精准度为：{100 * correct / total}%')