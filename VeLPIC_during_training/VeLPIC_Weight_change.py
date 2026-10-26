#!/usr/bin/env python
# coding: utf-8

# In[12]:


from MASC import angle_pytorch as angle
from CNN_code import cnn_create
from MAVC import MASV_pytorch as mavc
import os
import warnings
warnings.filterwarnings("ignore")
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import gc
import time
import shutil
import argparse
import copy 
from tqdm import tqdm
import pandas as pd


# In[2]:


def pca_load_mavc(path_pca,run,epoch_present,subspace_type,p_layer,pca_neg_avg_train,num_class):
    pca_class = []
    for class_projection in range(0,num_class,1):
        folder_pca=f'{path_pca}/Run_{run}/{epoch_present}'
        file_name=f'pca_train_{subspace_type}_{p_layer}_{class_projection}.pt'
        pca_class1= torch.load(f'{folder_pca}/{file_name}').cuda()

        if pca_neg_avg_train[class_projection]==True:
            pca_class1=-pca_class1

        pca_class.append(pca_class1.to(torch.float32))

    final_tensor = torch.cat(pca_class, dim=0)

    return final_tensor

def weight_change(dummy_model,final_tensor,withbias=False,type_network='MLP'):
    if type_network=='MLP' or type_network=='AlexNet' :
        if withbias:
            if type_network=='MLP':
                dict_withbias=copy.deepcopy(dummy_model.output.state_dict())
                dict_withbias['weight']=final_tensor
                dummy_model.output.load_state_dict(dict_withbias)
                
            else:
                dict_withbias=copy.deepcopy(dummy_model.fc3.state_dict())
                dict_withbias['weight']=final_tensor
                dummy_model.fc3.load_state_dict(dict_withbias)

        else:
            if type_network=='MLP':
                dict_withoutbias=copy.deepcopy(dummy_model.output.state_dict())
                dict_withoutbias['weight']=final_tensor
                dict_withoutbias['bias']=torch.zeros(dict_withoutbias['bias'].shape[0], 
                                                     dtype=torch.float32,
                                                     device='cuda')
                dummy_model.output.load_state_dict(dict_withoutbias)
            else:
                dict_withoutbias=copy.deepcopy(dummy_model.fc3.state_dict())
                dict_withoutbias['weight']=final_tensor
                dict_withoutbias['bias']=torch.zeros(dict_withoutbias['bias'].shape[0], 
                                                     dtype=torch.float32,
                                                     device='cuda')
                dummy_model.fc3.load_state_dict(dict_withoutbias)
                
    if type_network=='CNN':
        dict_withbias=copy.deepcopy(dummy_model.fc3.state_dict())
        dict_withbias['weight']=final_tensor
        dummy_model.fc3.load_state_dict(dict_withbias)
    return dummy_model



def accuracy_model(model,loader,data_type):
    model.to(dev)

    model.eval()

    temp_acc = 0
    train_acc = 0

    count = 0
    for idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(dev), labels.to(dev)
        model.zero_grad()
        output = model(inputs)
        count += len(labels)

        temp_acc += (torch.argmax(output, 1) == labels).float().sum().item()
    train_acc = (temp_acc/count) * 100
    print(f'\n {data_type} accuracy:{train_acc}, correct: {temp_acc}, total:{count}')

    return train_acc

def results_store(type_network):
    if type_network=='MLP' or type_network=='AlexNet':
            results = {
              'epoch': [],
              'Train_acc_withbias': [],
              'Train_acc_org_withbias': [],
               'Test_acc_withbias': [],
              'Train_acc_zerobias': [],
              'Train_acc_org_zerobias': [],
               'Test_acc_zerobias': [],
            }
    elif type_network=='CNN':
        results = {
          'epoch': [],
          'Train_acc_zerobias': [],
          'Train_acc_org_zerobias': [],
           'Test_acc_zerobias': [],
        }
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select model_type, datasets.")
    
    corrution_prob = [0.0,0.2,0.4,0.6,0.8,1.0]
    model_type = ['CNN','MLP','AlexNet']
    datasets = ['CIFAR10','MNIST','FashionMNIST','TinyImageNet']
    run_values=[1,2,3]

    parser.add_argument(
        "-corr", type=float, required=True, choices=corrution_prob, help="select corruption"
    )
    parser.add_argument(
        "-model", type=str, required=True, choices=model_type, help="select model_type"
    )
    parser.add_argument(
        "-dataset", type=str, required=True, choices=datasets, help="select dataset"
    )
    parser.add_argument(
        "-run", type=int, required=True, choices=run_values, help="select run"
    )
    args = parser.parse_args()

    # Access arguments
    corrupt = args.corr
    type_network = args.model
    ds = args.dataset
    run=args.run

    if corrupt not in corrution_prob:
        args.print_help()
    if type_network not in model_type:
        args.print_help()
    if ds not in datasets:
        args.print_help()
        
    if run not in run_values:
        args.print_help()

    t0 = time.time()
    torch.manual_seed(42)
        
#     n=0.99
    n=1
    runs = 4
    
    general_pca='/home/simrank/work/pca_during_training_angle/angle_results_1pc'
    results_folder=f'results/MAVC_weight_change/{ds}_{type_network}'
    pca_layer,data_layer,num_class=mavc.layer_name(type_network)
    
    p_layer=pca_layer[-1]
    d_layer=data_layer[-1]


#     for corrupt in corrution_prob:
    network_path,test_loader,corrupted_train,og_targets=cnn_create.original_dataset(type_network,ds,corrupt)
    true_labels=torch.from_numpy(og_targets).to(dev)

    path_pca=f'{general_pca}/{ds}_{type_network}/results_{corrupt}/pca'

#     for run in range(1,runs): 
    temp_path = f'VeLPIC_temp/Network_data_{corrupt}_{ds}_{type_network}_{run}'
    os.makedirs(temp_path,exist_ok=True)

    accuracy_path=f'{results_folder}/results_{corrupt}/Run_{run}'
    os.makedirs(accuracy_path,exist_ok=True)
    for epoch in cnn_create.selected_model_list(network_path,corrupt,run):
        print(epoch)
        results=results_store(type_network)
        dummy_model=cnn_create.model_create(type_network,ds)

        dummy_model.load_state_dict(torch.load(f'{network_path}/{corrupt}/Run_{run}/{epoch}',map_location =dev))
        cnn_create.saving_activations_lastlayer(
                        temp_path,dummy_model,corrupted_train,test_loader,
                        og_targets,dev,type_network)

        _,_,batch_size=cnn_create.model_params(type_network,ds,dummy_model)

        train_loader=mavc.relabels_dataset(corrupted_train,true_labels,batch_size)


        epoch_present=cnn_create.epochnumber(epoch)

        layer_output,_,_=mavc.data_loading_train(temp_path, d_layer)
        subspace_type='corrupt'
        class_dot=mavc.dot_part_train(p_layer,path_pca,epoch_present,
                                      run,subspace_type,layer_output,num_class)
        num_images=layer_output.shape[0]
        pca_neg_avg_train=mavc.avg_neg(class_dot,num_images,num_class=num_class)
        final_tensor=pca_load_mavc(path_pca,run,epoch_present,
                                   subspace_type,p_layer,pca_neg_avg_train,num_class)

        #code with bias not zero
        if type_network=='MLP' or type_network=='AlexNet':
            dummy_model=weight_change(dummy_model,final_tensor,withbias=True,type_network=type_network)
            train_acc=accuracy_model(dummy_model,corrupted_train,'corrupted train')
            train_acc_org=accuracy_model(dummy_model,train_loader,'original train')
            test_acc=accuracy_model(dummy_model,test_loader,'test')

            results['Train_acc_withbias'].append(train_acc)
            results['Train_acc_org_withbias'].append(train_acc_org)
            results['Test_acc_withbias'].append(test_acc)

        #code with bias = zero
        dummy_model=weight_change(dummy_model,final_tensor,withbias=False,type_network=type_network)

        train_acc=accuracy_model(dummy_model,corrupted_train,'corrupted train')
        train_acc_org=accuracy_model(dummy_model,train_loader,'original train')
        test_acc=accuracy_model(dummy_model,test_loader,'test')
        results['epoch'].append(epoch_present)
        results['Train_acc_zerobias'].append(train_acc)
        results['Train_acc_org_zerobias'].append(train_acc_org)
        results['Test_acc_zerobias'].append(test_acc)


        # Construct path
        csv_path = os.path.join(f"{accuracy_path}/Accuracy_MAVC_weight.csv")

        df = pd.DataFrame(data=results)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

        print(f"done till {epoch}")    
    shutil.rmtree(temp_path)


