import torch
import os
import pandas as pd
import numpy as np
import gc
from MASC import subspace_pytorch as subspace
from MASC import scratch_pca as sp
import pickle
import time
from torch.utils.data import TensorDataset, DataLoader,Subset
import copy
def layer_name(type_network):
    if type_network =='AlexNet':
        data_layer=['after_flatten','after_relu_fc1','after_relu_fc2'] #'input_layer',
        pca_layer=['flattern','fc1','fc2']#'input',
        num_class=200
    #cnn model
    if type_network =='CNN':
        data_layer=['input_layer','input_fc_0','output_fc_0_after_noise_relu',
                    'output_fc_1_after_noise_relu','output_fc_2_after_noise_relu']
        pca_layer=['input','flattern','fc1','fc2','fc3']
        num_class=10
    #mlp model
    if type_network =='MLP':
        data_layer=['input_layer','after_relu_fc1', 
                    'after_relu_fc2','after_relu_fc3', 
                    'after_relu_fc4']
        pca_layer=['input','fc1','fc2','fc3','fc4']
        num_class=10
        
    return pca_layer,data_layer,num_class


def data_loading_train(temp_path,d_layer):
    #training data
    with open(f'{temp_path}/{d_layer}_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)  

    layer_output = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    layer_output = torch.stack(layer_output).cuda()

    #original labels
    with open(f'{temp_path}/y_value_original_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)  

    og_targets = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    og_targets = torch.stack(og_targets).cuda()


    #corrupt labels
    with open(f'{temp_path}/y_value_corrupted_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)     

    cor_targets = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    cor_targets = torch.stack(cor_targets).cuda()

    
    return layer_output,og_targets,cor_targets

def data_loading_test(temp_path,d_layer):

    #testing images
    with open(f'{temp_path}/{d_layer}_test.pkl', 'rb') as file: 
        myvar = pickle.load(file) 

    layer_output_test = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    layer_output_test = torch.stack(layer_output_test).cuda()

     #original labels
    with open(f'{temp_path}/y_value_corrupted_test.pkl', 'rb') as file: 
        myvar = pickle.load(file)  
    original_test_labels = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]

    original_test_labels = torch.stack(original_test_labels).cuda()
    
    return layer_output_test,original_test_labels



def max_class_layer(class_dot, num_images, num_class=10):
    class_dot = class_dot.view(num_class, num_images)
    max_values, max_indices = torch.max(class_dot, dim=0)
    result = torch.stack([max_indices.float(), max_values], dim=1)
    return result

def accuracy_angle_layer(y_pred, y):
    score_l = 0
    for image_i in range(len(y)):
        if y_pred[image_i][0] == y[image_i]:
            score_l += 1
    score_l = round(score_l / len(y), 4)
    return score_l

def dot_part_train(p_layer,path_pca,epoch,run,subspace_type,layer_output,num_class):
    batch_size=100
    for class_projection in range(0,num_class,1):
        folder_pca=f'{path_pca}/Run_{run}/{epoch}'
        file_name=f'pca_train_{subspace_type}_{p_layer}_{class_projection}.pt'
        pca_class1= torch.load(f'{folder_pca}/{file_name}').cuda()

        #training data
        for i in range(0, layer_output.shape[0], batch_size):
            batch_output = layer_output[i:i + batch_size].to(torch.float64)
            batch_output=batch_output.to(torch.float64)
            pca_class1=pca_class1.to(torch.float64)
            data_pcadot = torch.matmul(batch_output,pca_class1.T)
            if i == 0:
                class_dot_batch = data_pcadot
            else:
                class_dot_batch = torch.cat((class_dot_batch, data_pcadot), dim=0)  

        if class_projection == 0:
            class_dot = class_dot_batch
        else:
            class_dot = torch.cat((class_dot, class_dot_batch), dim=0)
            
    return class_dot

def dot_part_test(p_layer,path_pca,epoch,run,subspace_type,layer_output_test,num_class):
    batch_size=100
    for class_projection in range(0,num_class,1):
        folder_pca=f'{path_pca}/Run_{run}/{epoch}'
        file_name=f'pca_train_{subspace_type}_{p_layer}_{class_projection}.pt'
        pca_class1= torch.load(f'{folder_pca}/{file_name}').cuda()

        # Project testing data onto classwise subspaces
        for j in range(0, layer_output_test.shape[0], batch_size):
            batch_output = layer_output_test[j:j + batch_size].to(torch.float64)
            batch_output=batch_output.to(torch.float64)
            pca_class1=pca_class1.to(torch.float64)
            data_pcadot = torch.matmul(batch_output,pca_class1.T)
            if j == 0:
                class_dot_test_batch = data_pcadot
            else:
                class_dot_test_batch = torch.cat((class_dot_test_batch, data_pcadot), dim=0) 

        if class_projection == 0:
            class_dot_test = class_dot_test_batch
        else:
            class_dot_test = torch.cat((class_dot_test, class_dot_test_batch), dim=0) 
            
    return class_dot_test




def MAVC_fn(type_network,temp_path,path_pca,run,results,n,epoch_present,subspace_type):
    print(type_network)
    pca_layer,data_layer,num_class=layer_name(type_network)
    results_angle1=results['angle_1']
    results_angle2=results['angle_2']
    
    epoch=epoch_present
    
    #results folders 
    os.makedirs(f'{results_angle1}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle1}/Run_{run}/{epoch}',exist_ok=True)
    path2=f'{results_angle1}/Run_{run}/{epoch}'

    os.makedirs(f'{results_angle2}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle2}/Run_{run}/{epoch}',exist_ok=True)
    path3=f'{results_angle2}/Run_{run}/{epoch}'
    

    with torch.no_grad(): 
        for p_layer,d_layer in zip(pca_layer,data_layer):

            layer_output,og_targets,cor_targets,layer_output_test,original_test_labels=data_loading(temp_path,d_layer)

            class_dot,class_dot_test=dot_part(p_layer,path_pca,epoch,run,subspace_type,layer_output,
                                              layer_output_test,num_class)
            
            if subspace_type=='corrupt':
            
                num_images=layer_output.shape[0]
                y_pred=max_class_layer(class_dot,num_images,num_class=num_class)

                #corrupted label
                acc_overall=accuracy_angle_layer(y_pred,cor_targets)

                filename='acc_overall_train'
                d = {'acc_overall':[acc_overall]}
                df1 = pd.DataFrame(data=d)
                df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 

                 #original labels
                acc_overall=accuracy_angle_layer(y_pred,og_targets)
                filename='acc_overall_train'
                d = {'acc_overall':[acc_overall]}
                df1 = pd.DataFrame(data=d)
                df1.to_csv(f"{path3}/layer_{p_layer}_{filename}.csv") 

                
            else:            
                num_images=layer_output.shape[0]
                y_pred=max_class_layer(class_dot,num_images,num_class=num_class)

                 #original labels
                acc_overall=accuracy_angle_layer(y_pred,og_targets)
                filename='acc_overall_train'
                d = {'acc_overall':[acc_overall]}
                df1 = pd.DataFrame(data=d)
                df1.to_csv(f"{path3}/layer_{p_layer}_{filename}.csv") 


            #test data
            num_images=layer_output_test.shape[0]
            y_pred=max_class_layer(class_dot_test,num_images,num_class=num_class)
            #true label test data
            acc_overall=accuracy_angle_layer(y_pred,original_test_labels)
            filename='acc_overall_test'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 

            del layer_output, og_targets,cor_targets,layer_output_test,original_test_labels
            gc.collect()
            torch.cuda.empty_cache()

            
def MAVC_fn_test(type_network,temp_path,path_pca,run,results,n,epoch_present,subspace_type):
    print(type_network)
    pca_layer,data_layer,num_class=layer_name(type_network)
    results_angle1=results['angle_1']    
    epoch=epoch_present
    
    #results folders 
    os.makedirs(f'{results_angle1}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle1}/Run_{run}/{epoch}',exist_ok=True)
    path2=f'{results_angle1}/Run_{run}/{epoch}'

    

    with torch.no_grad(): 
        for p_layer,d_layer in zip(pca_layer,data_layer):

            layer_output_test,original_test_labels=data_loading_test(temp_path,d_layer)

            class_dot_test=dot_part_test(p_layer,path_pca,epoch,run,subspace_type,
                                              layer_output_test,num_class)
            

            #test data
            num_images=layer_output_test.shape[0]
            y_pred=max_class_layer(class_dot_test,num_images,num_class=num_class)
            #true label test data
            acc_overall=accuracy_angle_layer(y_pred,original_test_labels)
            filename='acc_overall_test'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 

            del layer_output_test,original_test_labels
            gc.collect()
            torch.cuda.empty_cache()
            

def dot_part_test_npca(p_layer,path_pca,epoch,run,subspace_type,layer_output_test,num_class,pca_neg_avg_train):
    batch_size=100
    for class_projection in range(0,num_class,1):
        folder_pca=f'{path_pca}/Run_{run}/{epoch}'
        file_name=f'pca_train_{subspace_type}_{p_layer}_{class_projection}.pt'
        pca_class1= torch.load(f'{folder_pca}/{file_name}').cuda()

        if pca_neg_avg_train[class_projection]==True:
            pca_class1=-pca_class1
        
        # Project testing data onto classwise subspaces
        for j in range(0, layer_output_test.shape[0], batch_size):
            batch_output = layer_output_test[j:j + batch_size].to(torch.float64)
            batch_output=batch_output.to(torch.float64)
            pca_class1=pca_class1.to(torch.float64)
            data_pcadot = torch.matmul(batch_output,pca_class1.T)
            if j == 0:
                class_dot_test_batch = data_pcadot
            else:
                class_dot_test_batch = torch.cat((class_dot_test_batch, data_pcadot), dim=0) 

        if class_projection == 0:
            class_dot_test = class_dot_test_batch
        else:
            class_dot_test = torch.cat((class_dot_test, class_dot_test_batch), dim=0) 
            
    return class_dot_test

def avg_neg(class_dot,num_images,num_class=10):
    
    class_dot = class_dot.view(num_class, num_images)
    avg_class_dot=torch.mean(class_dot, dim=1)
    pca_neg_avg_train = [value.item() < 0 for value in avg_class_dot.cpu()]    

    return pca_neg_avg_train

def MAVC_fn_neg_pca(type_network,temp_path,path_pca,run,results,n,epoch_present,subspace_type):

#     print(type_network)
    pca_layer,data_layer,num_class=layer_name(type_network)
    results_angle1=results['angle_1']
    results_angle2=results['angle_2']

    epoch=epoch_present

    #results folders 
    os.makedirs(f'{results_angle1}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle1}/Run_{run}/{epoch}',exist_ok=True)
    path2=f'{results_angle1}/Run_{run}/{epoch}'

    os.makedirs(f'{results_angle2}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle2}/Run_{run}/{epoch}',exist_ok=True)
    path3=f'{results_angle2}/Run_{run}/{epoch}'

    with torch.no_grad(): 
        for p_layer,d_layer in zip(pca_layer,data_layer):

            layer_output,_,_=data_loading_train(temp_path, d_layer)

            class_dot=dot_part_train(p_layer,path_pca,epoch,run,subspace_type,layer_output,num_class)

            num_images=layer_output.shape[0]

            pca_neg_avg_train=avg_neg(class_dot,num_images,num_class=num_class)

            layer_output_test,original_test_labels=data_loading_test(temp_path,d_layer)
            class_dot_test=dot_part_test_npca(p_layer,path_pca,epoch,run,subspace_type,
                                              layer_output_test,num_class,pca_neg_avg_train)

            y_pred=max_class_layer(class_dot,num_images,num_class=num_class)

            #test data
            num_images=layer_output_test.shape[0]
            y_pred=max_class_layer(class_dot_test,num_images,num_class=num_class)
            #true label test data
            acc_overall=accuracy_angle_layer(y_pred,original_test_labels)
            filename='acc_overall_test'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 

            del layer_output,layer_output_test,original_test_labels
            gc.collect()
            torch.cuda.empty_cache()
            
            
# Custom dataset wrapper
class SubsetWithNewLabels(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, new_labels):
        self.dataset = dataset
        self.indices = indices
        self.new_labels = new_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _ = self.dataset[original_idx]  # Ignore original label
        return image, self.new_labels[idx]
    
def relabels_dataset(corrupted_train,y_train,batch_size):
    # Original dataset
    original_loader = copy.deepcopy(corrupted_train)

    # Indices for sub-loader (use the first 50 samples as an example)
    subset_indices = list(range(len(original_loader)))  

    # New labels (must be of the same length as subset_indices)
    new_labels = y_train.round().to(torch.int64).to('cpu')

    # Create the modified dataset and DataLoader
    subset_dataset = SubsetWithNewLabels(original_loader.dataset, subset_indices, new_labels)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    return subset_loader