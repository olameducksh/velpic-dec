import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms, models
import torch.optim as optim
from tqdm import tqdm

def load_model(ds_name='cifar10'):
    assert ds_name in ['cifar10', 'mnist']
    if ds_name == 'cifar10':
        num_classes = 10
        input_channels = 3
        input_size = 32
    elif ds_name == 'mnist':
        num_classes = 10
        input_channels = 1
        input_size = 28

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # Convolutional layer block 1
            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
            self.bn2 = nn.BatchNorm2d(64)
            
            # Max pooling layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
            
            # Convolutional layer block 2
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
            self.bn4 = nn.BatchNorm2d(128)
            
            # Fully connected layers
            flat_size = self._get_flat_size(input_size, input_channels)
            self.fc1 = nn.Linear(flat_size, 256) 
            self.fc2 = nn.Linear(256, num_classes)
            
            # Dropout layer
            self.dropout = nn.Dropout(0.5)

        def _get_flat_size(self, input_size, input_channels):
            """Compute the flattened size of the feature maps after the conv layers."""
            with torch.no_grad():
                dummy = torch.randn(1, input_channels, input_size, input_size)
                x = F.relu(self.bn1(self.conv1(dummy)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)  
                
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.relu(self.bn4(self.conv4(x)))
                x = self.pool(x)
                
                flat_size = x.view(1, -1).shape[1]

            return flat_size
        
        def forward(self, x):
            # Convolutional block 1
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)  # Downsample
            
            # Convolutional block 2
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)  # Downsample
            
        
            x = x.view(x.size(0), -1)
    
        
            # Fully connected block
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    model = SimpleCNN()
    print("Model loaded")


    return model

def load_data(ds='cifar10', test_only = False, batch_size=32):
    data_foldername='~/work/pca_during_training_angle/data'
    assert ds in ['cifar10', 'mnist']

    if ds == 'cifar10':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize( (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        
        train_dataset = datasets.CIFAR10(root=data_foldername, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root=data_foldername, train=False, transform=transform, download=True)

    elif ds == 'mnist':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize( (0.1307,), (0.3081,))
            ])
        
        train_dataset = datasets.MNIST(root=data_foldername, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=data_foldername, train=False, transform=transform, download=True)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if test_only:
        return test_dl
    else:
        return train_dl, test_dl


def get_ds_labels(ds_name):
    if ds_name == 'cifar10':
        return {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
    elif ds_name == 'mnist':
        return {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9'
        }
    
    elif ds_name == 'fashionmnist':
        return {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        }
    
# def denorm_img(x, ds_name):
#     if ds_name == 'cifar10':
#         mean=[0.4914, 0.4822, 0.4465]
#         std=[0.247, 0.243, 0.261]
        
#         if isinstance(mean, list):
#             mean = torch.tensor(mean)
#         if isinstance(std, list):
#             std = torch.tensor(std)

#     elif ds_name == 'mnist':
#         mean=[0.1307]
#         std=[0.3081]
#         if isinstance(mean, list):
#             mean = torch.tensor(mean)
#         if isinstance(std, list):
#             std = torch.tensor(std)
#     return x *  std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def denorm_img(x, ds_name):
    if ds_name == 'cifar10':
        mean=[0.5, 0.5, 0.5] 
        std=[0.5, 0.5, 0.5]
        
        if isinstance(mean, list):
            mean = torch.tensor(mean)
        if isinstance(std, list):
            std = torch.tensor(std)

    elif ds_name == 'mnist' or ds_name=='fashionmnist':
        mean=[0.5]
        std=[0.5]
        if isinstance(mean, list):
            mean = torch.tensor(mean)
        if isinstance(std, list):
            std = torch.tensor(std)
    return x *  std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def train_model(model, train_dl, test_dl, n_epochs=20, ds='cifar10', model_save_path='./'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    print("Starting training")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = model.to(device)

    for epoch in range(n_epochs):
        epoch_acc = 0
        for img, label in tqdm(train_dl):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            epoch_acc += (outputs.argmax(dim=1) == label).sum().item()
            loss.backward()
            optimizer.step()

        epoch_acc /= len(train_dl.dataset)
        print(f'Epoch {epoch}/{n_epochs} training accuracy: {epoch_acc}, validation accuracy: {get_test_acc(model, test_dl)}')
    print('Finished training')
    
    torch.save(model.state_dict(), f'{model_save_path}/cnn_{ds}_{n_epochs}epochs.pt')
    return model

def get_test_acc(model, test_dl):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for img, label in test_dl:
            img, label = img.to(device), label.to(device)
            output = model(img)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    return correct / len(test_dl.dataset)

def load_pretrained_model(path):
    if 'cifar10' in path:
        model = load_model(ds_name='cifar10')
    elif 'mnist' in path:
        model = load_model(ds_name='mnist')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(path))
    model = model.to(device)

    return model



    
if __name__ == '__main__':
    # model = load_model(ds_name='cifar10')
    # train_dl, test_dl = load_data(ds='cifar10', test_only=False, batch_size=32)
    # train_model(model, train_dl, test_dl, n_epochs=30, ds='cifar10', model_save_path='./model_weights')
    model = None
    model = load_model(ds_name='mnist')
    train_dl, test_dl = load_data(ds='mnist', test_only=False, batch_size=32)
    train_model(model, train_dl, test_dl, n_epochs=20, ds='mnist', model_save_path='./model_weights')