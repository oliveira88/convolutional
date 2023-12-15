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

PARAR = False
def handler(signum, frame):
    global PARAR
    print("\b\n[Ctrl-C pressionado. Parando o treinamento...]")
    PARAR = True

signal.signal(signal.SIGINT, handler)

# Download: https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
# mude o './cifar10/' pelo local onde descompactou seu train_data,mas deixe o 'train' eo 'test'
train_data = ImageFolder('./kaggle/train', transform=v2.Compose([ToTensor(), Normalize(0,1)]))
test_data = ImageFolder('./kaggle/test', transform=v2.Compose([ToTensor(), Normalize(0,1)]))
assert train_data.classes == test_data.classes, 'Voce deve deixar os subdiretorios de "train" e "test" iguais.'
print(f'Classes do train_data: {train_data.classes}')
print(f'Tamanho da figura: {train_data[0][0].shape}')

def show_example(img, label):
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f'Classe: {train_data.classes[label]} ({label})')
    plt.show()

show_example(*train_data[randint(0, len(train_data))])

# agora, vamor criar uns auxiliares que vão fornecer os dados em um laço de iteração
# se tiver pouca memória, reduza o batch_size abaixo:
batch_size = 32
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size)

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


net = Net()
print(net)

# E vamos escolher um auxiliar para mudar os pesos de nossa rede:
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# E, então, vamos treinar a rede:
max_epocas = 100
for epoch in range(max_epocas):  # loop sobre o dataset várias vezes
    print(f'Treinando epoca {epoch+1}:')
    total_loss, running_loss = 0.0, 0.0
    for i, data in enumerate(trainloader, 0):
        # obtem as entradas, 'data' é uma lista de [inputs, labels]
        inputs, labels = data

        # zerando os gradientes
        optimizer.zero_grad()

        # passando as imagens na rede, calculando os erros e atualizando a rede
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # imprimindo as estatísticas
        total_loss += loss.item()
        running_loss += loss.item()
        if i%100 == 99:
            print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/100:.3f}')
            running_loss = 0.0
    print(f'Loss: {total_loss/len(trainloader):.3f}')
    if PARAR: break

print('O treino acabou.')
print('Salvando o modelo em "model_5.pth"')
torch.save(net.state_dict(), "model_5.pth")

print('Testando:')
correct = 0
total = 0
# desativamos o cálculo dos gradientes, pois são necessários só no treinamento
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # passamos as imagens de testes pela rede para ver o resultado
        outputs = net(images)
        # a classe com maior valor é a escolhida na predição
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracia da rede: {100*correct/total:.2f} %')

# Para carregar a rede do arquivo, deve-se utilizar o seguinte código:
#net = Net()
#net.load_state_dict(torch.load("model_2.pth"))