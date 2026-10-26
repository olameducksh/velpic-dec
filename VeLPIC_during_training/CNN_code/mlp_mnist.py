# -*- coding: utf-8 -*-
from ast import Index
import numpy as np
from scipy import stats
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from tqdm import tqdm
import copy
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST


# from neurodl.targeted_neurogenesis import targeted_neurogenesis

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""**Load dataset**"""

def load_data(
    mode,
    data_folder="./data",
    num_workers=16,
    batch_size=50,
    split=0.1,
    seed=23,
    fashion=False,
    cifar10=False,
    mnist=False,
):
    """
    Helper function to read in image dataset, and split into
    training, validation and test sets.
    ===
    mode: str, ['validation', 'test]. If 'validation', training data
         will be divided based on split parameter.
         If test, .valid = None, and all training data is used for training
    split: float, where 0 < split < 1. Where train = split * num_samples
        and valid = (1 - split) * num_samples
    seed: int, random seed to generate validation/training split
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    assert mode in ["validation", "test"], "mode not validation nor test"

    if fashion:
        trainset = torchvision.datasets.FashionMNIST(
            data_folder,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        testset = torchvision.datasets.FashionMNIST(
            data_folder,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        print("Loaded FMNIST dataset")
    elif cifar10:
        trainset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True, transform=transform
        )
        

        testset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif mnist:
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(
            root=data_folder, train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.MNIST(
            root=data_folder, train=False, download=True, transform=transform
        )
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=data_folder, train=True, download=True, transform=transform
        )
        

        testset = torchvision.datasets.CIFAR100(
            root=data_folder, train=False, download=True, transform=transform
        )
    print('train len: ',len(trainset))
    print('test len: ',len(testset))

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    if mode == "validation":
        from sklearn.model_selection import train_test_split

        num_train = 50000
        indices = list(range(num_train))

        train_idx, valid_idx = train_test_split(
            indices, test_size=split, random_state=seed
        )

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler,
        )

        validloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=valid_sampler,
            drop_last=True,
        )
        print("Created data loaders")
        return trainloader, validloader, testloader

    elif mode == "test":
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )
        print("Created data loaders")
        return trainloader, testloader


class CIFAR100_data(object):
    def __init__(
        self,
        mode="validation",
        data_folder="./data",
        batch_size=50,
        fashion=False,
        num_workers=16,
        split=0.1,
        seed=23,
        cifar10=False,
    ):
        if mode == "validation":
            self.train, self.valid, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                batch_size=batch_size,
                num_workers=num_workers,
                split=split,
                fashion=fashion,
                cifar10=cifar10,
                seed=seed,
            )
        elif mode == "test":
            self.train, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                seed=seed,
                fashion=fashion,
                batch_size=batch_size,
                num_workers=num_workers,
                cifar10=cifar10,
            )
            self.valid = None

"""**Model**"""

class NgnCnn(nn.Module):
    def __init__(
        self,
        layer_size=250,
        channels=3,
        control=False,
        seed=0,
        excite=False,
        neural_noise=None,
    ):
        torch.manual_seed(seed)
        super(NgnCnn, self).__init__()
        # parameters
        self.ablate = False
        self.dropout = 0
        self.channels = channels
        self.excite = excite
        self.n_new = 0
        self.control = False
        if self.control:
            self.idx_control = np.random.choice(
                range(layer_size), size=8, replace=False
            )
        self.neural_noise = neural_noise

        # 3@16x16
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.layer_size = layer_size

        self.fc_new_in = nn.ModuleList()
        self.fc_new_out = nn.ModuleList()

        if self.channels == 3:
            self.cnn_output = 64 * 4 * 4
        elif self.channels == 1:
            self.cnn_output = 64 * 9
        # three fully connected layers
        self.fcs = nn.ModuleList(
            [
                nn.Linear(self.cnn_output, self.layer_size),  # 0
                nn.Linear(self.layer_size, self.layer_size),  # 1 on dim 2 neurogenesis
                nn.Linear(self.layer_size, self.layer_size),  # 2
            ]
        )
        self.fc3 = nn.Linear(self.layer_size, 10, bias=False)

    def forward(self, x, extract_layer=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.pool4(x)

        x = x.view(-1, self.cnn_output)

        for ix, fc in enumerate(self.fcs):
            x = fc(x)
            if self.neural_noise is not None and ix == 0 and self.training:
                mean, std = self.neural_noise
                noise = torch.zeros_like(x, device=dev)
                noise = noise.log_normal_(mean=mean, std=std)
                x = x * noise
            x = F.relu(x)

            if self.excite and ix == 1 and self.n_new and self.training:
                idx = self.idx_control if self.control else self.idx
                excite_mask = torch.ones_like(x)
                excite_mask[:, idx] = self.excite
                excite_mask.to(dev)
                x = x * excite_mask

            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = torch.renorm(x, 1, 1, 3)  # max norm

            # for ablation experiments
            if self.ablate:
                if ix == 1:
                    activation_size = x.size()[1]
                    if self.ablation_mode == "random":
                        ablate_size = int(self.ablation_prop * activation_size)
                        indices = np.random.choice(
                            range(activation_size),
                            size=size,
                            replace=False,
                        )
                    if self.ablation_mode == "targetted":
                        indices = self.ablate_indices
                    x[:, indices] = 0
            if extract_layer == ix:
                return x
        x = self.fc3(x)

        return x

    # def add_new(
    #     self,
    #     p_new=0.01,
    #     replace=True,
    #     targeted_portion=None,
    #     return_idx=False,
    #     layer=1,
    # ):
    #     """
    #     pnew: float, proportion of hidden layer to add
    #     replace: float, from 0-1 which is the proportion of new neurons that replace old neurons
    #     target: bool, neurons that are lost are randomly chosen, or targetted
    #             based on variance of activity
    #     """
    #     # get a copy of current parameters
    #     bias = [ix.bias.detach().clone().cpu() for ix in self.fcs]
    #     current = [ix.weight.detach().clone().cpu() for ix in self.fcs]
    #     if layer == 2:
    #         current_fc3 = self.fc3.weight.detach().clone().cpu()

    #     # how many neurons to add?
    #     if not p_new:
    #         return
    #     # if int given, use this as number of neurons to add
    #     if (p_new % 1) == 0:
    #         n_new = p_new
    #     # if float given, use to calculate number of neurons to add
    #     else:
    #         n_new = int(self.layer_size * p_new)

    #     if targeted_portion is not None:
    #         targ_diff = round(targeted_portion * current[layer].shape[0]) - n_new
    #         if targ_diff <= 0:
    #             n_new = n_new + targ_diff - 3

    #     self.n_new = n_new
    #     n_replace = n_new if replace else 0  # number lost
    #     difference = n_new - n_replace  # net addition or loss
    #     self.layer_size += difference  # final layer size

    #     # reallocate the weights and biases
    #     if replace:
    #         # if some neurons are being removed
    #         if targeted_portion is not None:
    #             try:
    #                 weights, mask = targeted_neurogenesis(
    #                     current[layer], n_replace, targeted_portion, self.training
    #                 )
    #             except ValueError:
    #                 print(
    #                     "n_replace",
    #                     n_replace,
    #                     "targ",
    #                     targeted_portion * (current[layer].shape[0]),
    #                 )

    #             # if neurons are targetted for removal
    #             idx = np.where(mask)[0]
    #             bias[1] = np.delete(bias[1], idx)
    #             current[layer] = np.delete(current[layer], idx, axis=0)
    #             current[layer + 1] = np.delete(current[layer + 1], idx, axis=1)
    #         else:
    #             # if neurons are randomly chosen for removal
    #             idx = np.random.choice(
    #                 range(current[layer].shape[0]), size=n_replace, replace=False
    #             )

    #             # delete idx neurons from bias and current weights (middle layer)
    #             bias[1] = np.delete(bias[1], idx)
    #             current[layer] = np.delete(current[layer], idx, axis=0)
    #             try:
    #                 current[layer + 1] = np.delete(current[layer + 1], idx, axis=1)
    #             except IndexError:
    #                 current_fc3 = np.delete(current_fc3, idx, axis=1)


    #         self.idx = idx

    #     # create new weight shapes
    #     w_in = torch.Tensor(
    #         self.layer_size,
    #         current[layer].shape[1],
    #     )
    #     b_in = torch.Tensor(self.layer_size)
    #     if layer < 2:
    #         w_out = torch.Tensor(
    #             current[layer + 1].shape[0],
    #             self.layer_size,
    #         )
    #     elif layer == 2:
    #         w_out = torch.Tensor(
    #             current_fc3.shape[0],
    #             self.layer_size,
    #         )

    #     # initialize new weights
    #     nn.init.kaiming_uniform_(w_in, a=math.sqrt(5))
    #     nn.init.kaiming_uniform_(w_out, a=math.sqrt(5))

    #     # in bias (out bias unaffected by neurogenesis)
    #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w_in)
    #     bound = 1 / math.sqrt(fan_in)
    #     nn.init.uniform_(b_in, -bound, bound)

    #     # put back current bias and weights into newly initiliazed layers
    #     b_in[:-n_new] = bias[1]
    #     w_in[:-n_new, :] = current[layer]
    #     if layer == 2:
    #         w_out[:, :-n_new] = current_fc3
    #     else:
    #         w_out[:, :-n_new] = current[layer + 1]

    #     # create the parameters again
    #     self.fcs[layer].bias = nn.Parameter(b_in)
    #     self.fcs[layer].weight = nn.Parameter(w_in)
    #     if layer == 2:
    #         self.fc3.weight = nn.Parameter(w_out)
    #     else:
    #         self.fcs[layer + 1].weight = nn.Parameter(w_out)

    #     # need to send all the data to GPU again
    #     self.fcs.to(dev)
    #     if layer == 2:
    #         self.fc3.to(dev)

    #     if return_idx and (n_replace > 0):
    #         return idx

############### corrupted data loader and labels

class AlexNet(nn.Module):

  def __init__(self, classes=100):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, classes),
    )
    self.fc1 = nn.Linear(256 * 1 * 1, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, classes)
    
    
    self.input_img = None
    self.after_flatten = None
    self.before_relu_fc1 = None
    self.after_relu_fc1 = None
    self.before_relu_fc2 = None
    self.after_relu_fc2 = None
    self.before_relu_fc3 = None
    self.after_relu_fc3 = None

  def forward(self, x):
    self.input_img = x.clone().detach()
    x = self.features(x)
    x = torch.flatten(x, 1)
    #x = self.classifier(x)
    
    self.after_flatten = x.clone().detach()
    x = self.fc1(x)
    self.before_relu_fc1 = x.clone().detach()
    x = F.relu(x)
    self.after_relu_fc1 = x.clone().detach()
    
    x = self.fc2(x)
    self.before_relu_fc2 = x.clone().detach()
    x = F.relu(x)
    self.after_relu_fc2 = x.clone().detach()
    
    x = self.fc3(x)
    self.before_relu_fc3 = x.clone().detach()
    #x = F.relu(x)
    #self.after_relu_fc3 = x.clone().detach()
    
    return x

  def get_intermediate_states(self):
    return [
    self.input_img,
    self.after_flatten,
    self.after_relu_fc1,
    self.after_relu_fc2,
    self.before_relu_fc3    
    ]

class mlp(nn.Module):

  def __init__(self, classes=10,mnist=True):
    super(mlp, self).__init__()

    if mnist:
        self.fc1 = nn.Linear(28*28,128)
    else:
        self.fc1 = nn.Linear(32*32*3,128)
    self.fc2 = nn.Linear(128,512)
    self.fc3 = nn.Linear(512,2048)
    self.fc4 = nn.Linear(2048,2048)
    self.output = nn.Linear(2048, classes)    
    
    self.input_img = None
    self.after_flatten = None
    self.before_relu_fc1 = None
    self.after_relu_fc1 = None
    self.before_relu_fc2 = None
    self.after_relu_fc2 = None
    self.before_relu_fc3 = None
    self.after_relu_fc3 = None
    self.before_relu_fc4 = None
    self.after_relu_fc4 = None

  def forward(self, x):
    self.input_img = x.clone().detach()
    x = torch.flatten(x, 1)
    #x = self.classifier(x)
    
    self.after_flatten = x.clone().detach()
    x = self.fc1(x)
    self.before_relu_fc1 = x.clone().detach()
    x = F.relu(x)
    self.after_relu_fc1 = x.clone().detach()
    
    x = self.fc2(x)
    self.before_relu_fc2 = x.clone().detach()
    x = F.relu(x)
    self.after_relu_fc2 = x.clone().detach()
    
    x = self.fc3(x)
    self.before_relu_fc3 = x.clone().detach()
    x = F.relu(x)
    self.after_relu_fc3 = x.clone().detach()
    
    x = self.fc4(x)
    self.before_relu_fc4 = x.clone().detach()
    x = F.relu(x)
    self.after_relu_fc4 = x.clone().detach()
    
    x = self.output(x)
    
    return x

  def get_intermediate_states(self):
    return [
    self.input_img,
    self.after_flatten,
    self.after_relu_fc1,
    self.after_relu_fc2,
    self.after_relu_fc3,
    self.after_relu_fc4,
    ]


class CIFAR100Corrupted(CIFAR100):
    def __init__(self, corrupt_prob, num_classes=100, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets

class CIFAR10Corrupted(CIFAR10):
    def __init__(self, corrupt_prob, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets

class FashionMNISTCorrupted(FashionMNIST):
    def __init__(self, corrupt_prob, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets

class MNISTCorrupted(MNIST):
    def __init__(self, corrupt_prob, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets



def get_cifar_dataloaders_corrupted(corrupt_prob=0, batch_size=50,fashion=False,cifar10=False,mnist=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if fashion:
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        trainset = FashionMNISTCorrupted(root='./data', train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        testset = FashionMNISTCorrupted(root='./data', train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)  
    elif cifar10:
        trainset = CIFAR10Corrupted(root='./data', train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        testset = CIFAR10Corrupted(root='./data', train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)
    elif mnist:
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        trainset = MNISTCorrupted(root='./data', train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        testset = MNISTCorrupted(root='./data', train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)
    else:
        trainset = CIFAR100Corrupted(root='./data', train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        testset = CIFAR100Corrupted(root='./data', train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)
    og_prob, og_targets, cor_targets = trainset.get_targets()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, og_prob, og_targets, cor_targets

############### corrupted data loader and labels


"""**Training loop**"""
if __name__ == "__main__":

    torch.manual_seed(42)
    batch_size = 32

    _, test_loader = load_data("test",num_workers=1,batch_size=batch_size,mnist=True)

    # original_train_labels = []
    original_test_labels = []
    # corrupt_train_labels = []
    # corrupt_test_labels = []
    #
    # for _,y in train_loader:
    #     original_train_labels.append(y)
    #
    for _,y in test_loader:
        original_test_labels.append(y)
    #
    # corrupt_train_loader =
    #
    # for _, y in train_loader:
    #     original_train_labels.append(y)

    corrution_prob = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] #0.05, 0.1, 0.9, 0.95, 0.99,

    runs = 4

    path = 'MLP_MNIST'
    os.makedirs(path,exist_ok=True)
    network_path = os.path.join(path,'Network')
    os.makedirs(network_path, exist_ok=True)
    res_path = os.path.join(path,'Accuracy_results')
    for corrupt in corrution_prob:
        cur_network_dir = os.path.join(network_path, str(corrupt))
        # temp2 = os.path.join(cur_dir, 'Accuracy_results')
        os.makedirs(cur_network_dir, exist_ok=True)
        corrupted_train, _ , prob, og_targets, cor_targets = get_cifar_dataloaders_corrupted(corrupt, batch_size=batch_size,mnist=True)
        if prob == corrupt:
            mask = og_targets == cor_targets
            temp_count = np.sum(mask)
            percent = (temp_count/len(mask))*100
            percent = round(percent,2)
            label_data = {
                'True Label': og_targets,
                'Corrupted Label': cor_targets,
            }
            label_csv = pd.DataFrame(data=label_data)
            label_csv.to_csv(os.path.join(cur_network_dir,f'label_info_corruption_{corrupt}_percent_{percent}.csv'))
        for run in range(1,runs):
          run_path = os.path.join(cur_network_dir,f'Run_{run}')
          os.makedirs(run_path,exist_ok=True)

          results = {
              'epoch': [],
              'Train_accuracy': [],
              'Test_accuracy': [],
          }

          model = mlp(mnist=True)
          #model = AlexNet()

          #optimizer = torch.optim.Adam(model.parameters(),lr=0.0002)
          #optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
          #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,momentum=0.9, weight_decay=1e-2)
          optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,momentum=0.9)
          torch.save(model.state_dict(),os.path.join(run_path,'./initialized_model.pth'))

          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

          print(f"device: {device}")

          model.to(device)

          loss_func = nn.CrossEntropyLoss()

          epoches = 500

          for epoch in range(epoches):
            train_loss = 0
            test_loss = 0
            train_acc = 0
            test_acc = 0
            temp_acc = 0
            count = 0
            
            for idx, (inputs, labels) in tqdm(enumerate(corrupted_train), total=len(corrupted_train), desc="Mini Batches"):
              inputs, labels = inputs.to(device), labels.to(device)
              model.zero_grad()

              output = model(inputs)
              loss = loss_func(output, labels)
              loss.backward()
              train_loss += loss.item()
              temp_acc += (torch.argmax(output, 1) == labels).float().sum().item()
              count += len(labels)

              # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
              optimizer.step()
            train_acc = (temp_acc/count) * 100

            print(f'\n Corrupt: {corrupt}, Run: {run} Epoch: {epoch} Training accuracy:{train_acc}, correct: {temp_acc}, total:{count}')
            results['Train_accuracy'].append(train_acc)
            results['epoch'].append(epoch+1)

            acc = 0
            test_count = 0
            total_loss = 0
            batches = 0
            model.eval()
            with torch.no_grad():
              for i, (x_batch, y_batch) in tqdm(
                  enumerate(test_loader), total=len(test_loader), desc="Validation Round"
              ):
                  x_batch = x_batch.to(device)
                  y_batch = y_batch.to(device)
                  y_pred = model(x_batch)
                  loss = loss_func(y_pred, y_batch)
                  total_loss += loss.item()
                  acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()
                  test_count += len(y_batch)
                  batches = i + 1

            test_acc = (acc / test_count) * 100
            results['Test_accuracy'].append(test_acc)
            model.train()
            print(f'\n Corrupt: {corrupt}, Run: {run} Epoch: {epoch} Testing accuracy:{test_acc}, correct: {acc}, total:{test_count}')
            torch.save(model.to('cpu').state_dict(), os.path.join(run_path,f'model_{epoch}.pth'))
            model.to(device)

            if train_acc == 100 or train_acc == 100.0 :
              break
          temp_res_path = os.path.join(res_path,str(corrupt))
          os.makedirs(temp_res_path, exist_ok=True)
          df = pd.DataFrame(data=results)
          df.to_csv(os.path.join(temp_res_path,f"Run_{run}.csv"))

        del corrupted_train