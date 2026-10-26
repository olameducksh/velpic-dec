#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/home/simrank/work/Adverserial_Attack')

from MASC import angle_pytorch_attack as angle
from CNN_code import cnn_create
from adv_attacks_main import attacks as attack_code
from adv_attacks_main import model_utils
import os
import warnings
warnings.filterwarnings("ignore")
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import gc
import time
from tqdm import tqdm
import torch.nn as nn
import argparse 
import pandas as pd
import shutil
import re

# In[2]:


def create_folder(results_folder,name_type,eps):
    results={}
    os.makedirs(f'{results_folder}/results_{eps}_{name_type}',exist_ok=True)
    os.makedirs(f'{results_folder}/results_{eps}_{name_type}/angle_results',exist_ok=True)
    results['angle_1']=f'{results_folder}/results_{eps}_{name_type}/angle_results'
    #results pca
    os.makedirs(f'{results_folder}/results_{eps}_{name_type}/pca',exist_ok=True)
    results['pca']=f'{results_folder}/results_{eps}_{name_type}/pca' 
    
    os.makedirs(f'{results_folder}/results_{eps}_{name_type}',exist_ok=True)
    results['accuracy']=f'{results_folder}/results_{eps}_{name_type}'     
    return results

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', '1'}:
        return True
    elif value.lower() in {'false', 'no', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def model_name(path,run,corrupt):

    folder_path = f"{path}/{corrupt}/Run_{run}"
    model_files = [f for f in os.listdir(folder_path) if f.startswith("model_") and f.endswith(".pth")]
    model_numbers = [(f, int(re.search(r"model_(\d+).pth", f).group(1))) for f in model_files]
    model_number = max(model_numbers, key=lambda x: x[1])[0]

    return os.path.join(folder_path,model_number)
 
def path_model(type_network,data_type):
    if type_network=='MLP':
        if data_type =='MNIST':
            network_path = '/mnt/NVME2TB/simran/PCA/MLP_MNIST/Network'
            ds_name='mnist'
        else:
            network_path = '/mnt/NVME2TB/simran/PCA/MLP_CIFAR10/Network'
            ds_name='cifar10'
    if type_network=='CNN':
        if data=='CIFAR10':
            network_path = '/mnt/8TB/simran/PCA/CIFAR_10_Wodrop/Network'
            ds_name='cifar10'
        elif data=='FashionMNIST':
            network_path = '/mnt/NVME2TB/simran/PCA/FashionMNIST/Network'
            ds_name='fashionmnist'
        else :
            network_path = '/mnt/NVME2TB/simran/PCA/MNIST/Network'
            ds_name='mnist' 
    return network_path,ds_name


def data_set_fn_noise(data_type):
    if data_type=='MNIST':
        corrupted_train_noise, _ , _, og_targets, _ =cnn_create.get_cifar_dataloaders_corrupted(
                corrupt_prob=corrupt,batch_size=1,mnist=True,noise=noise)
        _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,mnist=True)
        corrupted_train, _ , _, og_targets, _ =cnn_create.get_cifar_dataloaders_corrupted(
            corrupt_prob=corrupt,batch_size=1,mnist=True)

    if data_type=='CIFAR10':
        corrupted_train_noise, _ , _, og_targets, _ =cnn_create.get_cifar_dataloaders_corrupted(
                corrupt_prob=corrupt,batch_size=1,cifar10=True,noise=noise)
        _, test_loader , _,_,_=cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,cifar10=True)
        corrupted_train, _ , _, og_targets, _ =cnn_create.get_cifar_dataloaders_corrupted(
            corrupt_prob=corrupt,batch_size=1,cifar10=True)

    if data_type=='FashionMNIST':
        corrupted_train_noise, _ , _, og_targets, _ =cnn_create.get_cifar_dataloaders_corrupted(
                corrupt_prob=corrupt,batch_size=1, fashion=True,noise=noise)
        _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(
            batch_size=1,fashion=True)
        corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(
            corrupt_prob=corrupt,batch_size=1,fashion=True)
    return corrupted_train,test_loader,og_targets,corrupted_train_noise


def data_set_fn(data_type):
    if data_type=='MNIST':
        _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,mnist=True)
        corrupted_train, _ , _, og_targets, _ =cnn_create.get_cifar_dataloaders_corrupted(
            corrupt_prob=corrupt,batch_size=1,mnist=True)

    if data_type=='CIFAR10':
        _, test_loader , _,_,_=cnn_create.get_cifar_dataloaders_corrupted(batch_size=1,cifar10=True)
        corrupted_train, _ , _, og_targets, _ =cnn_create.get_cifar_dataloaders_corrupted(
            corrupt_prob=corrupt,batch_size=1,cifar10=True)

    if data_type=='FashionMNIST':
        _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(
            batch_size=1,fashion=True)
        corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(
            corrupt_prob=corrupt,batch_size=1,fashion=True)
    return corrupted_train,test_loader,og_targets
    





if __name__ == "__main__":

    
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Select dataset, model,eps, n_value and noise_flag")
    datasets = ['MNIST','CIFAR10','FashionMNIST','CIFAR100','Tiny_ImageNet']
    model_types=['CNN','MLP','AlexNet']
    # 1 will be used for 1/255, 2 for 2/255 and 510 for 510/255=2
    epsilions = [1.0,2.0,2.55,127.5, 510.0,12.75,25.5,51.0,76.5,204.0,255.0]
    n_values =[0.99,0.80]#1.0,0.50,

    parser.add_argument(
        "-dataset", type=str, required=True, choices=datasets, help="select dataset"
    )
    parser.add_argument(
        "-model", type=str, required=True, choices=model_types, help="select model type"
    )
    
    parser.add_argument(
        "-eps", type=float, required=True, choices=epsilions, help="select epsilion value"
    )
    
    args = parser.parse_args()

    # Access arguments
    data = args.dataset
    type_network = args.model
    eps_value=args.eps
    if data not in datasets:
        args.print_help()
    if type_network not in model_types:
        args.print_help()   
    if eps_value not in epsilions:
        args.print_help()


    runs = 4
    corrupt=0.0
    for iters_value in [10,40]:
        for n in n_values:
            t0 = time.time()
            #path for temprary activation storage and data loading
            temp_path = f'PGDNetwork_data_{data}_{eps_value}_{type_network}_{corrupt}_{n}_{iters_value}'

            attack_params ={'eps': eps_value/255, 'targeted': False,
                            'iters' : iters_value,
                            'step_size' : 0.1}
            results_folder=f'Results/PGD_results/data_{iters_value}/{data}_{type_network}/{corrupt}/{n}'


            corrupted_train,test_loader,og_targets=data_set_fn(data)

            os.makedirs(temp_path,exist_ok=True)
            os.makedirs(results_folder,exist_ok=True)


            network_path,ds_name=path_model(type_network,data)
            #results folders
            results_org=create_folder(results_folder,'original',eps_value)
            results_pgd=create_folder(results_folder,'pgd',eps_value)

            # model accuracy for fgsm testing data 
            accuracy_pgd=[]
            for run in range(1,runs): 
                t_data=time.time()
                if type_network=='CNN':
                    if data=='CIFAR10':
                        model = cnn_create.NgnCnn()
                    else:
                        model = cnn_create.NgnCnn(channels=1) 
                if type_network=='MLP':
                    if data=='CIFAR10':
                        model = cnn_create.mlp(mnist=False)
                    else:
                        model = cnn_create.mlp(mnist=True)          

                file_name=model_name(network_path,run,0.0)
                model.load_state_dict(torch.load(file_name))
                model.to(device)
                #calculating accuracy fo fgsm attack testing data
                acc=attack_code.eval_attack(model, test_loader, 
                                            attack_params, attack_algorithm='pgd', ds_name=ds_name)
                accuracy_pgd.append(acc)

                # loader for attack
                pgd_loader = attack_code.create_pgd_loader(test_loader, model, attack_params)

                model.to('cpu')

                cnn_create.loading_saving_activations(temp_path,model,corrupted_train,
                                                      test_loader,og_targets,device,type_network)
                cnn_create.loading_saving_activations_attack(temp_path,model,
                                                             pgd_loader,og_targets,device,'pgd',type_network)
                del model

                #Angle work #testing data
                angle.angle_work_corrupt(type_network,temp_path,run,results_org,n,'trained_model','original')
                #testing attack data
                angle.angle_work_corrupt(type_network,temp_path,run,results_pgd,n,'trained_model','pgd')   
                t_data2 = time.time()
                t_data_time = t_data2-t_data
                print(f"time taken : {t_data_time} sec run {run} done")
            path_temp=results_pgd['accuracy']  
            d = {'Test_accuracy':accuracy_pgd}
            df2 = pd.DataFrame(data=d)
            df2.to_csv(f"{path_temp}/Accuracy.csv") 

            shutil.rmtree(temp_path)

            t1 = time.time()
            total = t1-t0
            print(f'total time taken {total} sec')

