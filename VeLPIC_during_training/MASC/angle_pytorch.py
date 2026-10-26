import torch
import os
import pandas as pd
import numpy as np
import gc
from MASC import subspace_pytorch as subspace
from MASC import scratch_pca as sp
import pickle
import time
def angle(X_output_layer, X_pca_new):
    layer_angle = []
    for layer in range(len(X_pca_new)):
        dot_product = torch.diag(torch.matmul(X_output_layer[layer], X_pca_new[layer].T))
        norm_X_output = torch.norm(X_output_layer[layer], dim=1)
        norm_X_pca_new = torch.norm(X_pca_new[layer], dim=1)
        cos_theta = dot_product / (norm_X_output * norm_X_pca_new)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure cos_theta is within the valid range
        layer_angle.append(torch.acos(cos_theta) * (180.0 / torch.pi))
    return torch.stack(layer_angle)

def angle_layer(X_output_layer, X_pca_new):
    dot_product = torch.diag(torch.matmul(X_output_layer, X_pca_new.T))
    norm_X_output = torch.norm(X_output_layer, dim=1)
    norm_X_pca_new = torch.norm(X_pca_new, dim=1)
    cos_theta = dot_product / (norm_X_output * norm_X_pca_new)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    layer_angle = torch.acos(cos_theta) * (180.0 / torch.pi)
    return layer_angle

def least_class(class_angle, num_images, number_class=10):
    temp_final = []
    for layer in range(class_angle.shape[0]):
        temp = [[torch.tensor(200000.0), torch.tensor(200000.0)] for _ in range(num_images)]
        for j in range(num_images):
            for k in range(number_class):
                num = k * num_images + j
                if class_angle[layer][num] < temp[j][1]:
                    temp[j] = [k, class_angle[layer][num]]
        temp_final.append(temp)
    return temp_final

# def least_class_layer(class_angle, num_images, number_class=10):
#     temp = [[torch.tensor(200000.0), torch.tensor(200000.0)] for _ in range(num_images)]
#     for j in range(num_images):
#         for k in range(number_class):
#             num = k * num_images + j
#             if class_angle[num] < temp[j][1]:
#                 temp[j] = [k, class_angle[num]]
#     return temp





def least_class_layer(class_angle, num_images, number_class=10):
#     tin = time.time()
    # Reshape class_angle to (number_class, num_images) for efficient comparison
    class_angle = class_angle.view(number_class, num_images)
    
    # Find the minimum angle and its corresponding class index for each image
    min_values, min_indices = torch.min(class_angle, dim=0)
    
    # Stack the results together (class index and the corresponding minimum angle)
    result = torch.stack([min_indices.float(), min_values], dim=1)
#     tout = time.time()
#     total = tout-tin
#     print(f'total time taken {total} sec inside least_class_layer function')
    return result

def accuracy_angle(y_pred, y):
    score_f = []
    for layer in range(len(y_pred)):
        score_l = 0
        for image_i in range(len(y)):
            if y_pred[layer][image_i][0] == y[image_i]:
                score_l += 1
        score_f.append(round(score_l / len(y), 4))
    return score_f
def accuracy_angle_layer(y_pred, y):
#     tin = time.time()
    score_l = 0
    for image_i in range(len(y)):
        if y_pred[image_i][0] == y[image_i]:
            score_l += 1
    score_l = round(score_l / len(y), 4)
#     tout = time.time()
#     total = tout-tin
#     print(f'total time taken {total} sec inside accuracy_angle_layer function')
    return score_l
def acc_class_angle(y_pred, y, number_class=10):
#     tin = time.time()
    score_f = []
    for class_i in range(number_class):
        score_c = []
        for layer in range(len(y_pred)):
            score_l = 0
            count = 0
            for image_i in range(len(y)):
                if y[image_i] == class_i:
                    count += 1
                    if y_pred[layer][image_i][0] == y[image_i]:
                        score_l += 1
            if count > 0:
                score_c.append(round(score_l / count, 4))
        score_f.append(score_c)
#     tout = time.time()
#     total = tout-tin
#     print(f'total time taken {total} sec inside acc_class_angle function')
    return score_f

def acc_class_angle_layer(y_pred, y, number_class=10):
#     tin = time.time()
    score_f = []
    for class_i in range(number_class):
        score_l = 0
        count = 0
        for image_i in range(len(y)):
            if y[image_i] == class_i:
                count += 1
                if y_pred[image_i][0] == y[image_i]:
                    score_l += 1
        if count > 0:
            score_f.append(round(score_l / count, 4))
            
#     tout = time.time()
#     total = tout-tin
#     print(f'total time taken {total} sec inside acc_class_angle_layer function')
    return score_f


def acc_class_angle_layer2(y_pred, y, number_class=10):
#     tin = time.time()
    # Assuming y_pred is a 2D tensor with predictions in the first column
    y_pred = y_pred[:, 0]  # Extract predicted labels

    # Initialize a tensor to store class-wise accuracies
    score_f = torch.zeros(number_class, dtype=torch.float32, device=y.device)

    # Calculate accuracy for each class
    for class_i in range(number_class):
        mask = y == class_i  # Mask for samples belonging to the current class
        total = mask.sum().item()  # Total samples in this class
        if total > 0:
            correct = (y_pred[mask] == y[mask]).sum().float()  # Correct predictions for this class
            accuracy = correct / total  # Calculate accuracy
            score_f[class_i] = torch.round(accuracy * 10000) / 10000  # Round to 4 decimals

#     tout = time.time()
#     total = tout-tin
#     print(f'total time taken {total} sec inside acc_class_angle_layer2 function')
    return score_f.detach().cpu().tolist()


def angle_work_corrupt(type_network,temp_path,run,results,n,epoch_present):
#     print('inside angle_work_corrupt')
    t0 = time.time()
    #AlexNet model
    if type_network =='AlexNet':
        data_layer_name=['after_flatten','after_relu_fc1',
                     'y_value_corrupted']
        #'input_layer','after_relu_fc2',
        data_output_name=['flattern','fc1']#'input','fc2'
        num_class=200
    #cnn model
    #cnn model
    if type_network =='CNN':
        data_layer_name=['input_layer','input_fc_0','output_fc_0_after_noise_relu',
                         'output_fc_1_after_noise_relu']# 'output_fc_2_after_noise_relu'
        data_output_name=['input','flattern','fc1','fc2'] #'fc3'
        num_class=10

    #mlp model        
    if type_network =='MLP':
        data_layer_name=['after_relu_fc4','y_value_corrupted'] #'input_layer','after_relu_fc1','after_relu_fc2', 'after_relu_fc3',
        data_output_name=['fc4']#'input','fc1','fc2','fc3',
        num_class=10
        
    results_angle1=results['angle_1']
    results_angle2=results['angle_2']
    results_pca=results['pca']
    batch_size=100
    epoch=epoch_present
    #results folders 
    os.makedirs(f'{results_angle1}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle1}/Run_{run}/{epoch}',exist_ok=True)
    path2=f'{results_angle1}/Run_{run}/{epoch}'

    os.makedirs(f'{results_angle2}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle2}/Run_{run}/{epoch}',exist_ok=True)
    path3=f'{results_angle2}/Run_{run}/{epoch}'
    
    os.makedirs(f'{results_pca}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_pca}/Run_{run}/{epoch}',exist_ok=True)
    path2_pca=f'{results_pca}/Run_{run}/{epoch}' 
    
    with torch.no_grad(): 
        for data,data_out in zip(data_layer_name,data_output_name):
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

#                 print("class_projection: ",class_projection,"X_pca_class: ",X_ppca.shape[0],
#                       "X_pca_values:", X_pca_values)
                torch.save(X_ppca.cpu(),f"{path2_pca}/pca_train_corrupt_{data_out}_{class_projection}.pt")
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
                df1.to_csv(f"{path2_pca}/pca_train_corrupt_{data_out}_var.csv")

            d = {'number_pca_components_class':X_pca_class
                    }
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2_pca}/pca_train_corrupt_{data_out}.csv")

            num_images=layer_output.shape[0]
            y_pred=least_class_layer(class_angle,num_images,number_class=num_class)

            #corrupted label
            acc_overall=accuracy_angle_layer(y_pred,cor_targets)
#             acc_class=acc_class_angle_layer2(y_pred,cor_targets,number_class=num_class)
            filename='acc_overall_train'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/layer_{data_out}_{filename}.csv") 
#             filename='acc_classwise_train'
#             d = {'acc_class':acc_class}
#             df2 = pd.DataFrame(data=d)
#             df2.to_csv(f"{path2}/layer_{data_out}_{filename}.csv")  

            #original labels
            acc_overall=accuracy_angle_layer(y_pred,og_targets)
#             acc_class=acc_class_angle_layer2(y_pred,og_targets,number_class=num_class)
            filename='acc_overall_train'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path3}/layer_{data_out}_{filename}.csv") 
#             filename='acc_classwise_train'
#             d = {'acc_class':acc_class}
#             df2 = pd.DataFrame(data=d)
#             df2.to_csv(f"{path3}/layer_{data_out}_{filename}.csv") 

            #test data
            num_images=layer_output_test.shape[0]
            y_pred=least_class_layer(class_angle_test,num_images,number_class=num_class)
            #true label test data
            acc_overall=accuracy_angle_layer(y_pred,original_test_labels)
#             acc_class=acc_class_angle_layer2(y_pred,original_test_labels,number_class=num_class)
            filename='acc_overall_test'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/layer_{data_out}_{filename}.csv") 
#             filename='acc_classwise_test'
#             d = {'acc_class':acc_class}
#             df2 = pd.DataFrame(data=d)
#             df2.to_csv(f"{path2}/layer_{data_out}_{filename}.csv")
            
            del layer_output, og_targets,cor_targets,layer_output_test,original_test_labels,obj
            gc.collect()
            torch.cuda.empty_cache()
    t1 = time.time()
    total = t1-t0
    print(f'total time taken {total} sec inside angle_work_corrupt function')
            
            
def angle_work_original(type_network,temp_path,run,results,n,epoch_present):
#     print("inside angle_work_original")
    #AlexNet model
    if type_network =='AlexNet':
        data_layer_name=['after_flatten','after_relu_fc1',
                     'y_value_corrupted']
        #'input_layer','after_relu_fc2',
        data_output_name=['flattern','fc1']#'input','fc2'
        num_class=200
    #cnn model
    if type_network =='CNN':
        data_layer_name=['input_layer','input_fc_0','output_fc_0_after_noise_relu',
                         'output_fc_1_after_noise_relu']# 'output_fc_2_after_noise_relu'
        data_output_name=['input','flattern','fc1','fc2'] #'fc3'
        num_class=10
    #mlp model
    if type_network =='MLP':
        
        data_layer_name=['input_layer','after_relu_fc1','after_relu_fc2',
                         'after_relu_fc3','y_value_corrupted'] #'after_relu_fc4','
        
        data_output_name=['input','fc1','fc2','fc3']#'fc4'
        num_class=10

    results_angle1=results['angle_1']
    results_angle2=results['angle_2']
    results_pca=results['pca']
    batch_size=100
    epoch=epoch_present
    #results folders 
    os.makedirs(f'{results_angle1}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle1}/Run_{run}/{epoch}',exist_ok=True)
    path2=f'{results_angle1}/Run_{run}/{epoch}'

    os.makedirs(f'{results_angle2}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle2}/Run_{run}/{epoch}',exist_ok=True)
    path3=f'{results_angle2}/Run_{run}/{epoch}'
    
    os.makedirs(f'{results_pca}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_pca}/Run_{run}/{epoch}',exist_ok=True)
    path2_pca=f'{results_pca}/Run_{run}/{epoch}' 
    
    with torch.no_grad(): 
        for data,data_out in zip(data_layer_name,data_output_name):
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
            obj=subspace.classwise_data(layer_output,og_targets,number_class=num_class)
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
                torch.save(X_ppca.cpu(),f"{path2_pca}/pca_train_original_{data_out}_{class_projection}.pt")
#                 print("class_projection: ",class_projection,"X_pca_class: ",X_ppca.shape[0],
#                      "percent_var:", percent_var)
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
                df1.to_csv(f"{path2_pca}/pca_train_original_{data_out}_var.csv")
                
            d = {'number_pca_components_class':X_pca_class
                    }
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2_pca}/pca_train_original_{data_out}.csv")


            num_images=layer_output.shape[0]
            y_pred=least_class_layer(class_angle,num_images,number_class=num_class) 

            #original labels
            acc_overall=accuracy_angle_layer(y_pred,og_targets)
#             acc_class=acc_class_angle_layer2(y_pred,og_targets,number_class=num_class)
            filename='acc_overall_train'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path3}/layer_{data_out}_{filename}.csv") 
#             filename='acc_classwise_train'
#             d = {'acc_class':acc_class}
#             df2 = pd.DataFrame(data=d)
#             df2.to_csv(f"{path3}/layer_{data_out}_{filename}.csv") 

            #test data
            num_images=layer_output_test.shape[0]
            y_pred=least_class_layer(class_angle_test,num_images,number_class=num_class)
            #true label test data
            acc_overall=accuracy_angle_layer(y_pred,original_test_labels)
#             acc_class=acc_class_angle_layer2(y_pred,original_test_labels,number_class=num_class)
            filename='acc_overall_test'
            d = {'acc_overall':[acc_overall]}
            df1 = pd.DataFrame(data=d)
            df1.to_csv(f"{path2}/layer_{data_out}_{filename}.csv") 
#             filename='acc_classwise_test'
#             d = {'acc_class':acc_class}
#             df2 = pd.DataFrame(data=d)
#             df2.to_csv(f"{path2}/layer_{data_out}_{filename}.csv")
            
            del layer_output, og_targets,cor_targets,layer_output_test,original_test_labels,obj
            gc.collect()
            torch.cuda.empty_cache()
            
def angle_misclassified(type_network,temp_path,run,results,epoch_1,epoch_2):
    # AlexNet model
    if type_network =='AlexNet':
        data_layer_name=['after_flatten','after_relu_fc1',
                     'y_value_corrupted']
        #'input_layer','after_relu_fc2',
        data_output_name=['flattern','fc1']#'input','fc2'
        num_class=200
    #cnn model
    if type_network =='CNN':
        data_layer_name=['input_layer','input_fc_0','output_fc_0_after_noise_relu', 'output_fc_1_after_noise_relu']# 'output_fc_2_after_noise_relu'
        data_output_name=['input','flattern','fc1','fc2'] #'fc3'
        num_class=10
    #mlp model
    if type_network =='MLP':
        
        data_layer_name=['input_layer','after_relu_fc1','after_relu_fc2',  'after_relu_fc3','y_value_corrupted'] #'after_relu_fc4','
        
        data_output_name=['input','fc1','fc2','fc3']#'fc4'
        num_class=10

    results_angle=results

    epochs = [epoch_1, epoch_2]
    batch_size=100

    n_values=[0.20,0.50,0.70,0.99]
    # results folders 
    
    os.makedirs(f'{results_angle}/Run_{run}',exist_ok=True)
    os.makedirs(f'{results_angle}/Run_{run}/{epoch_1}_{epoch_2}',exist_ok=True)
    path2=f'{results_angle}/Run_{run}/{epoch_1}_{epoch_2}'

    with torch.no_grad(): 
        for epoch in epochs:
            for data,data_out in zip(data_layer_name,data_output_name):
                #training data
                with open(f'{temp_path}/{data}_train_{epoch}.pkl', 'rb') as file: 
                    myvar = pickle.load(file)  

                layer_output = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
                layer_output = torch.stack(layer_output).cuda()

                #original labels
                with open(f'{temp_path}/y_value_original_train.pkl', 'rb') as file: 
                    myvar = pickle.load(file)  

                og_targets = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
                og_targets = torch.stack(og_targets).cuda()


                #corrupt labels
                with open(f'{temp_path}/y_value_corrupted_train_{epoch}.pkl', 'rb') as file: 
                    myvar = pickle.load(file)     

                cor_targets = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
                cor_targets = torch.stack(cor_targets).cuda()

                #testing images
                with open(f'{temp_path}/{data}_test_{epoch}.pkl', 'rb') as file: 
                    myvar = pickle.load(file) 

                layer_output_test = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
                layer_output_test = torch.stack(layer_output_test).cuda()

                 #original labels
                with open(f'{temp_path}/y_value_corrupted_test_{epoch}.pkl', 'rb') as file: 
                    myvar = pickle.load(file)  

                original_test_labels = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]

                original_test_labels = torch.stack(original_test_labels).cuda()
                obj=subspace.classwise_data(layer_output,cor_targets,number_class=num_class)

                for n in n_values:
                    for class_projection in range(0,num_class,1):
                        X_output_layer = torch.tensor(obj[f'class{class_projection}']).cuda()
                        X_output_layer_add = subspace.Added_data_layer(X_output_layer)
                        X_ppca = torch.from_numpy(sp.PCA_new_layer(X_output_layer_add.cpu(), n)).cuda().to(torch.float64)

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
                            class_angle_test = class_angle_test_batch
                        else:
                            class_angle_test = torch.cat((class_angle_test, class_angle_test_batch), dim=0)   

                    label_angle={'label':[],'angle_with_label':[]}
#                     print(original_test_labels.shape, class_angle_test.shape)
                    
                    for i in range(original_test_labels.shape[0]):
#                         print(f"original: {original_test_labels[i]}, angle of original: {class_angle_test[(i*10)+original_test_labels[i]]} ")
                   
                        label_angle['label'].append(original_test_labels[i].detach().cpu().item())
                        label_angle['angle_with_label'].append(class_angle_test[(i*10)+original_test_labels[i]].detach().cpu().item())

                    filename=f'acc_overall_test'
                    df1 = pd.DataFrame(data=label_angle)
                    df1.to_csv(f"{path2}/layer_{data_out}_{filename}_{n}_{epoch}.csv") 

                del layer_output, og_targets,cor_targets,layer_output_test,original_test_labels,obj
                gc.collect()
                torch.cuda.empty_cache()