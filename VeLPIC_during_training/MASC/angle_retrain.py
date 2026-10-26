import torch
import os
import pandas as pd
import numpy as np
import gc
from MASC import subspace_pytorch as subspace
from MASC import scratch_pca as sp
import copy
from torch.utils.data import TensorDataset, DataLoader,Subset
from tqdm import tqdm


from MASC.angle_pytorch import *

def peak_info(peak_epoch):
    if peak_epoch==1:# random initialization
        epoch='initialized_model.pth'
        interval=99
    if peak_epoch==10: # load standard training model at epoch 10
        epoch='model_9.pth'
        interval=90
          #without once_relabel
#     interval=5

    if peak_epoch==40:
        epoch='model_39.pth'
        interval=60

    
    return epoch,interval

def results_folder(type_network,ds,run,corrupt,data_out,load_epoch=10,n=1,once_relabel=False,model=False):
    if once_relabel:
        folder_relabel='once_relabel'
    else:
        folder_relabel='every_fifth'
    
    if n==1:
        angle='angle_results' 
    else:
        angle='angle_results_99' 
    
    if model:
        pred_model='model'
    else:
        pred_model='masc'

    b_path=f'MASC_retraining/{angle}/{folder_relabel}'
    results=f'{b_path}/{load_epoch}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}/{data_out}'
    os.makedirs(f'{results}',exist_ok=True)

    b_path=f'/mnt/SSD2TB/simran/PCA/MASC_retraining/{angle}/{folder_relabel}'
    run_path=f'{b_path}/{load_epoch}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}/{data_out}'
    os.makedirs(f'{run_path}',exist_ok=True)

    temp_path=f'temp_folder/{pred_model}_{angle}_{folder_relabel}_{load_epoch}_{type_network}_{ds}_{data_out}_{corrupt}'
    os.makedirs(f'{temp_path}',exist_ok=True)

    return results,run_path,temp_path



def results_model(type_network,ds,run,corrupt,load_epoch=10,n=1,once_relabel=False,model=False):
    if once_relabel:
        folder_relabel='once_relabel'
    else:
        folder_relabel='every_fifth'
    
    if n==1:
        angle='angle_results' 
    else:
        angle='angle_results_99' 
    
    if model:
        pred_model='model'
    else:
        pred_model='masc'

    b_path=f'MASC_retraining/{angle}/{folder_relabel}'
    results_model=f'{b_path}/{load_epoch}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}'
    os.makedirs(f'{results_model}',exist_ok=True)
    
    b_path=f'/mnt/SSD2TB/simran/PCA/MASC_retraining/{angle}/{folder_relabel}'
    run_model_path=f'{b_path}/{load_epoch}/{type_network}_{ds}/{corrupt}/Run_{run}/{pred_model}'
    os.makedirs(f'{run_model_path}',exist_ok=True)

    return results_model,run_model_path

def angle_work_corrupt_retrain(type_network,temp_path,results,n,epoch,data,data_out,num_class):

    batch_size=100
    #results folders 
    path2=f'{results}/{epoch}'
    os.makedirs(path2,exist_ok=True)
    
    
    with torch.no_grad(): 
        #training data
        with open(f'{temp_path}/{data}_train.pkl', 'rb') as file: 
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

        #testing images
        with open(f'{temp_path}/{data}_test.pkl', 'rb') as file: 
            myvar = pickle.load(file) 

        layer_output_test = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
        layer_output_test = torch.stack(layer_output_test).cuda()

         #original labels
        with open(f'{temp_path}/y_value_corrupted_test.pkl', 'rb') as file: 
            myvar = pickle.load(file)  

        original_test_labels = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]

        original_test_labels = torch.stack(original_test_labels).cuda()
        obj=subspace.classwise_data(layer_output,cor_targets,number_class=num_class)
        X_pca_class=[]
        X_pca_values=[]
        for class_projection in range(0,num_class,1):
            X_output_layer = torch.tensor(obj[f'class{class_projection}']).cuda()
            X_output_layer_add = subspace.Added_data_layer(X_output_layer)

            if n>=1:
                pca_vectors,percent_var=sp.PCA_new_layer(X_output_layer_add.cpu(), n)
                X_pca_values.append(percent_var)
            else:
                pca_vectors=sp.PCA_new_layer(X_output_layer_add.cpu(), n)

            X_ppca = torch.from_numpy(pca_vectors).cuda().to(torch.float64)

            X_pca_class.append(X_ppca.shape[0])

            torch.save(X_ppca.cpu(),f"{path2}/pca_train_corrupt_{data_out}_{class_projection}.pt")
            # Project training data onto classwise subspaces
            for i in range(0, layer_output.shape[0], batch_size):
                batch_output = layer_output[i:i + batch_size].to(torch.float64)
                X_pca_new = subspace.pca_change_layer(batch_output, X_ppca)
                layer_angle = angle_layer(batch_output, X_pca_new)
                if i == 0:
                    class_angle_batch = layer_angle
                else:
                    class_angle_batch = torch.cat((class_angle_batch, layer_angle), dim=0)            

            # Project testing data onto classwise subspaces
            for j in range(0, layer_output_test.shape[0], batch_size):
                batch_output = layer_output_test[j:j + batch_size].to(torch.float64)
                X_pca_new = subspace.pca_change_layer(batch_output, X_ppca)
                layer_angle = angle_layer(batch_output, X_pca_new)
                if j == 0:
                    class_angle_test_batch = layer_angle
                else:
                    class_angle_test_batch = torch.cat((class_angle_test_batch, layer_angle), dim=0) 

            if class_projection == 0:
                class_angle = class_angle_batch
                class_angle_test = class_angle_test_batch
            else:
                class_angle = torch.cat((class_angle, class_angle_batch), dim=0)
                class_angle_test = torch.cat((class_angle_test, class_angle_test_batch), dim=0)   




        if n>=1:
            d = {'pca_variance_class':X_pca_values}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/pca_train_corrupt_{data_out}_var.csv")
        else:
            d = {'number_pca_components_class':X_pca_class
                    }
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/pca_train_corrupt_{data_out}.csv")

        num_images=layer_output.shape[0]
        y_pred=least_class_layer(class_angle,num_images,number_class=num_class)

        #corrupted label
        acc_overall=accuracy_angle_layer(y_pred,cor_targets)
        acc_overall_org=accuracy_angle_layer(y_pred,og_targets)
        
        num_images=layer_output_test.shape[0]
        y_pred_test=least_class_layer(class_angle_test,num_images,number_class=num_class)
        #true label test data
        acc_overall_test=accuracy_angle_layer(y_pred_test,original_test_labels)

        filename='acc_overall'
        d = {'acc_overall_relabel':[acc_overall],
             'acc_overall_original':[acc_overall_org],
             'acc_overall_test':[acc_overall_test]
            }
        
        df1 = pd.DataFrame(data=d)
        df1.to_csv(f"{path2}/layer_{data_out}_{filename}.csv") 


        del layer_output, og_targets,cor_targets,layer_output_test,original_test_labels,obj
        gc.collect()
        torch.cuda.empty_cache()
    return y_pred[:,0],y_pred_test[:,0]

def layer_name(type_network):
    if type_network =='AlexNet':
        data_layer_name=['after_flatten','after_relu_fc1','after_relu_fc2',
                     'y_value_corrupted']    #'input_layer',
        data_output_name=['flattern','fc1','fc2']    #'input'
        num_class=200
    #cnn model
    if type_network =='CNN':
        data_layer_name=['input_fc_0','output_fc_0_after_noise_relu',
                         'output_fc_1_after_noise_relu','output_fc_2_after_noise_relu']#'input_layer',
        data_output_name=['flattern','fc1','fc2','fc3'] #'input',
        num_class=10
    #mlp model
    if type_network =='MLP':
        data_layer_name=['after_relu_fc1','after_relu_fc2', 
                         'after_relu_fc3','after_relu_fc4','y_value_corrupted'] #'input_layer',
        data_output_name=['fc1','fc2','fc3','fc4']#'input',
        num_class=10
    return data_layer_name,data_output_name,num_class



def data_saving(path, loader, dummy_model, name, dev, type_network, layername):
    dummy_model.to(dev)
    dummy_model.eval()

    layer_outputs = []
    y_values = []

    for x, y in loader:
        x = x.to(dev)
        _ = dummy_model(x)
        intermediate_states = dummy_model.get_intermediate_states()

        # Collect outputs according to the network type and layername
        if type_network == 'AlexNet':
            data_map = {
                'after_relu_fc1': intermediate_states[1][0].cpu(),
                'after_relu_fc2': intermediate_states[2][0].cpu(),
                'after_flatten': intermediate_states[3][0].cpu(),
            }
        elif type_network == 'CNN':
            data_map = {
                'input_layer': x.cpu().reshape(-1),
                'input_fc_0': intermediate_states[-2]['input_fc_0'][0].cpu(),
                'output_fc_0_after_noise_relu': intermediate_states[-2]['output_fc_0_after_noise_relu'][0].cpu(),
                'output_fc_1_after_noise_relu': intermediate_states[-2]['output_fc_1_after_noise_relu'][0].cpu(),
                'output_fc_2_after_noise_relu': intermediate_states[-2]['output_fc_2_after_noise_relu'][0].cpu(),
            }
        elif type_network == 'MLP':
            data_map = {
                'input_layer': x.cpu().reshape(-1),
                'after_relu_fc1': intermediate_states[2][0].cpu(),
                'after_relu_fc2': intermediate_states[3][0].cpu(),
                'after_relu_fc3': intermediate_states[4][0].cpu(),
                'after_relu_fc4': intermediate_states[5][0].cpu(),
            }
        else:
            raise ValueError("Unsupported network type.")

        # Save the output only if the layername is valid
        if layername not in data_map:
            raise ValueError(f"Layer '{layername}' not found in {type_network}.")

        layer_outputs.append(data_map[layername])
        y_values.append(y[0])

    # Save the selected layer output
    with open(f'{path}/{layername}_{name}.pkl', 'wb') as f:
        pickle.dump(layer_outputs, f)

    # Always save the labels
    with open(f'{path}/y_value_corrupted_{name}.pkl', 'wb') as f:
        pickle.dump(y_values, f)
        
def data_saving_layerwise(temp_path, corrupted_train,og_targets,test_loader, dummy_model,
                          dev, type_network, data):
    data_saving(temp_path, corrupted_train, dummy_model, 'train', dev, type_network, data)
    #saving original labels
    with open(f'{temp_path}/y_value_original_train.pkl', 'wb') as file: 
        pickle.dump(og_targets, file)
    data_saving(temp_path, test_loader, dummy_model, 'test', dev, type_network, data)
    
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

def training_process(subset_loader,train_loader,test_loader,
                     model,loss_func,optimizer,run_path,accuracy_path,dev,epoch_number,interval
                     ):
    
    results = {
      'epoch': [],
      'Train_accuracy': [],
      'Train_accuracy_org': [],
       'Test_accuracy': [],
    }


    model.to(dev)
    
    for epoch in range(epoch_number,epoch_number+interval):
        model.train()
        
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0
        temp_acc = 0
        count = 0

        for idx, (inputs, labels) in tqdm(enumerate(subset_loader), total=len(subset_loader),desc="Training"):
            inputs, labels = inputs.to(dev), labels.to(dev)
            model.zero_grad()
            output = model(inputs)
            loss = loss_func(output, labels.type(torch.int64))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += len(labels)

            temp_acc += (torch.argmax(output, 1) == labels).float().sum().item()
        train_acc = (temp_acc/count) * 100

        print(f'\n  Epoch: {epoch},Training accuracy:{train_acc}, correct: {temp_acc}, total:{count}')
        results['Train_accuracy'].append(train_acc)
        results['epoch'].append(epoch+1)
        
        
        model.eval()
        
        acc = 0
        test_count = 0
        total_loss = 0
        batches = 0
        
        with torch.no_grad():
            for i, (x_batch, y_batch) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc="Training original"):
                x_batch = x_batch.to(dev)
                y_batch = y_batch.to(dev)
                y_pred = model(x_batch)
                loss = loss_func(y_pred, y_batch.type(torch.int64))
                total_loss += loss.item()
                acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()

                test_count += len(y_batch)
                batches = i + 1

        test_acc = (acc / test_count) * 100
        results['Train_accuracy_org'].append(test_acc)
        print(f'\n Epoch: {epoch} training accuracy original:{test_acc}, correct: {acc}, total:{test_count}')
        
        
        acc = 0
        test_count = 0
        total_loss = 0
        batches = 0
        with torch.no_grad():
            for i, (x_batch, y_batch) in tqdm(
                enumerate(test_loader), total=len(test_loader), desc="Testing Round"):
                x_batch = x_batch.to(dev)
                y_batch = y_batch.to(dev)
                y_pred = model(x_batch)
                loss = loss_func(y_pred, y_batch.type(torch.int64))
                total_loss += loss.item()
                acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()

                test_count += len(y_batch)
                batches = i + 1

        test_acc = (acc / test_count) * 100
        results['Test_accuracy'].append(test_acc)
        print(f'\n Epoch: {epoch} Testing accuracy:{test_acc}, correct: {acc}, total:{test_count}')
        
        
        torch.save(model.to('cpu').state_dict(),
                       os.path.join(run_path,f'model_{epoch}.pth'))
        
        model.to(dev)

        
    # Construct path
    csv_path = os.path.join(f"{accuracy_path}/Accuracy_retrain.csv")

    # Convert current results to DataFrame
    df = pd.DataFrame(data=results)

    # If the CSV exists, append without header
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
        
    print(f"retraining done till {epoch}")
    
    return model