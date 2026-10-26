import torch
import torch.nn.functional as F
import torch.nn as nn


from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from model_utils import load_pretrained_model, load_data
from attacks import eval_attack, visualize_attack


def eps_acc_plot(model, test_ds, attack_params, attack_algorithm='fgsm', ds_name='cifar10'):
    if ds_name == 'cifar10': epsilons = np.arange(0, 0.5, 0.05)
    elif ds_name == 'mnist': epsilons = np.arange(0, 5, 0.5)
    # epsilons = [0]

    adv_accs = []
    for eps in epsilons:
        attack_params['eps'] = eps
        adv_acc = eval_attack(model, test_ds, attack_params, attack_algorithm, ds_name=ds_name)
        adv_accs.append(adv_acc)
        print(f"Eps: {eps}, Accuracy: {adv_acc}")
    plt.figure(figsize=(8, 8))
    plt.plot(epsilons, adv_accs)
    plt.xlabel('Epsilon')
    plt.ylabel('Adversarial accuracy')
    plt.title(f'{attack_algorithm.upper()} Adversarial accuracy vs epsilon')
    plt.savefig(f'./example_imgs/{ds_name}_{attack_algorithm}_acc_vs_eps.png')

if __name__ == '__main__':
    # torch seed
    torch.manual_seed(12)
    ds = 'mnist'
    model = load_pretrained_model('./model_weights/cnn_mnist_20epochs.pt')
    test_ds = load_data(ds, test_only=True)
    # attack_params ={'eps': 2, 'targeted': False} #* Sample FGSM attack params
    attack_params ={'eps': 2, 'targeted': False, 'iters': 40, 'step_size': 0.01} #* SamplePGD attack params
    # visualize_attack(model, test_ds, attack_params, attack_algorithm='fgsm', ds_name=ds)
    eps_acc_plot(model, test_ds, attack_params, attack_algorithm='pgd', ds_name=ds)

   