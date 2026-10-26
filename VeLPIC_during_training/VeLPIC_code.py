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



def results_fol_name(results_folder,corrupt,only_test=False):
    
    if  only_test==True:
        #results foldersfor corrupted subspace
        results_corr={}
        os.makedirs(f'{results_folder}/results_{corrupt}',exist_ok=True)
        os.makedirs(f'{results_folder}/results_{corrupt}/angle_results',exist_ok=True)
        results_corr['angle_1']=f'{results_folder}/results_{corrupt}/angle_results'

        #results folders for original subspace
        results_org={}
        os.makedirs(f'{results_folder}/results_{corrupt}/angle_results_exp3',exist_ok=True)
        results_org['angle_1']=f'{results_folder}/results_{corrupt}/angle_results_exp3'
    
    else:
        #results foldersfor corrupted subspace
        results_corr={}
        os.makedirs(f'{results_folder}/results_{corrupt}',exist_ok=True)
        os.makedirs(f'{results_folder}/results_{corrupt}/angle_results',exist_ok=True)
        results_corr['angle_1']=f'{results_folder}/results_{corrupt}/angle_results'
        #original training labels + corrupted training subspaces : exp2
        os.makedirs(f'{results_folder}/results_{corrupt}/angle_results_exp2',exist_ok=True)
        results_corr['angle_2']=f'{results_folder}/results_{corrupt}/angle_results_exp2'

        #results folders for original subspace
        results_org={}
        os.makedirs(f'{results_folder}/results_{corrupt}/angle_results_exp3',exist_ok=True)
        results_org['angle_1']=f'{results_folder}/results_{corrupt}/angle_results_exp3'
        results_org['angle_2']=f'{results_folder}/results_{corrupt}/angle_results_exp3'

    return results_corr,results_org

def fun_results(type_network,ds,corrupt):
    
    path_pca=f'../pca_during_training_angle/angle_results_1pc/{ds}_{type_network}/results_{corrupt}/pca'

    #path for temprary activation storage
    temp_path = f'Network_data/VeLPIC_{corrupt}_{ds}_{type_network}'
    os.makedirs(temp_path,exist_ok=True)
    results_folder=f'results/MAVC_angle_results_1pc_train/{ds}_{type_network}'
    os.makedirs(results_folder,exist_ok=True)
    return path_pca,temp_path,results_folder
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")
    
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

    if ds=='TinyImageNet':
        tiny=True
    else:
        tiny=False
    torch.manual_seed(42)
        
#     n=0.99
    n=1
#     runs = 4
    
#     for corrupt in corrution_prob:
    network_path,test_loader,corrupted_train,og_targets=cnn_create.original_dataset(type_network,ds,corrupt)
    path_pca,temp_path,results_folder=fun_results(type_network,ds,corrupt)

    results_corr,results_org=results_fol_name(results_folder,corrupt)

#     for run in range(1,runs): 
    for epoch in cnn_create.selected_model_list(network_path,corrupt,run,tiny=tiny):
        print(epoch)
        dummy_model=cnn_create.model_create(type_network,ds)

        dummy_model.load_state_dict(torch.load(f'{network_path}/{corrupt}/Run_{run}/{epoch}',map_location =dev))

        cnn_create.loading_saving_activations(
            temp_path,dummy_model,corrupted_train,test_loader,
            og_targets,dev,type_network)

        del dummy_model
        epoch_present=cnn_create.epochnumber(epoch)

        mavc.MAVC_fn_neg_pca(type_network,temp_path,path_pca,run,results_corr,n,epoch_present,subspace_type='corrupt')


    print(f"run {run} done")
    print(f'corrupt {corrupt} done')

    shutil.rmtree(temp_path)












