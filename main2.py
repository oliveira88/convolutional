import signal
from random import randint
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

device = torch.device("cuda")
# Agora, vamos criar nossa rede neural convolucional:
class Net(nn.Module):
    def __init__(self, numero_de_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, numero_de_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

batch_size = 32
train_data = ImageFolder('./kaggle/train', transform=v2.Compose([ToTensor(), Normalize(0,1)]))
test_data = ImageFolder('./kaggle/test', transform=v2.Compose([ToTensor(), Normalize(0,1)]))
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size)


net = Net()
net2 = Net()
net3 = Net()
net.load_state_dict(torch.load("model_2.pth"))
net2.load_state_dict(torch.load("model_3.pth"))
net3.load_state_dict(torch.load("model_4.pth"))

net.eval()
net2.eval()
net3.eval()

print('Testando:')
correct1 = 0
total1 = 0
correct2 = 0
total2 = 0
correct3 = 0
total3 = 0
# desativamos o cálculo dos gradientes, pois são necessários só no treinamento
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # passamos as imagens de testes pela rede para ver o resultado
        outputs1 = net(images)
        outputs2 = net2(images)
        outputs3 = net3(images)
        # a classe com maior valor é a escolhida na predição
        _, predicted1 = torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        _, predicted3 = torch.max(outputs3.data, 1)
        total1 += labels.size(0)
        total2 += labels.size(0)
        total3 += labels.size(0)
        correct1 += (predicted1 == labels).sum().item()
        correct2 += (predicted2 == labels).sum().item()
        correct3 += (predicted3 == labels).sum().item()

rates = [[100*correct1/total1], [100*correct2/total2], [100*correct3/total3]]

# print(f'Accuracia da rede 1: {rates[0]:.2f} %')
# print(f'Accuracia da rede 2: {rates[1]:.2f} %')
# print(f'Accuracia da rede 3: {rates[2]:.2f} %')
plt.boxplot(rates, labels=['Treino 1', 'Treino 2', 'Treino 3'])
plt.xlabel('Treinamento')
plt.ylabel('Taxa de Sucesso (%)')
plt.title('Boxplot das taxas de sucesso dos treinamentos')
plt.show()
