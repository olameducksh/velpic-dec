import scratch_pca as sp
import numpy as np

from sklearn.preprocessing import normalize

#extracting features from each layer for the data
def classwise_data(X,y,number_class=10):
    #dict holding the classes dataset
    obj={}
    for i in range(0,number_class):
        obj['class'+str(i)]=[]

    for i in range(len(y)):
        obj['class'+str(y[i])].append(X[i])
    return obj



def extract_layersfeatures(new_model,data):
    inp = new_model.input                                           # input placeholder
    outputs = [layer.output for layer in new_model.layers]          # all layer outputs
    functors=[]
    #input 
    X_output_layer=[data]
    #layers
    for out in outputs:
        functors.append(K.function([inp], [out]))
    for fun in functors:
        X_output_layer.append(fun([data])[0])
    return X_output_layer

def extract_layers(main_model, starting_layer_ix, ending_layer_ix):
    model_temp = Sequential()
    for ix in range(starting_layer_ix, ending_layer_ix):
        curr_layer = main_model.get_layer(index=ix)
        # copy this layer over to the new model
        model_temp.add(curr_layer)
    return model_temp

def cal_accuracy(y,output_pca,output_original):
    output_pca=np.argmax(output_pca,axis=1)
    output_original=np.argmax(output_original,axis=1)
    score_pca =0
    score_original=0

    for i in range(len(y)):
        if y[i]==output_pca[i]:
            score_pca=score_pca+1
        if y[i]==output_original[i]:
            score_original=score_original+1
    return score_pca/len(y),score_original/len(y)

def acc_layerwise_pcafeatures_originaldata(new_model,X_pca_try,X_output_layer,y):
    input_pca_shape=[]
    input_original_shape=[]
    output_pca_shape=[]
    output_original_shape=[]
    acc_pca_layer=[]
    acc_original_layer=[]    
    rankoriginal=[]
    ranklayerpca=[]
    for i in range(len(X_pca_try)):
        model_temp=extract_layers(new_model, i,len(X_pca_try)-1)
        
        input_original=X_output_layer[i]
        rankoriginal.append(np.linalg.matrix_rank(input_original))
        input_original_shape.append(input_original.shape)
        output_original=model_temp(input_original)
        output_original_shape.append(output_original.shape)
        
        input_pca = X_pca_try[i]
        ranklayerpca.append(np.linalg.matrix_rank(input_pca))
        input_pca_shape.append(input_pca.shape)
        output_pca=model_temp(input_pca)
        output_pca_shape.append(output_pca.shape)
        
        acc_pca,acc_original=cal_accuracy(y,output_pca,output_original)
        acc_pca_layer.append(acc_pca)
        acc_original_layer.append(acc_original)
        
    return input_pca_shape,input_original_shape,rankoriginal,ranklayerpca,output_pca_shape,output_original_shape,acc_pca_layer,acc_original_layer

# add data code
def add_neg(arr):
    return (-1 * arr)

def Added_data(X_outputadd_layer):
    X_new_final=[]
    for j in range(len(X_outputadd_layer)):
        X_newa=[]
        for i in range(X_outputadd_layer[j].shape[0]):
            X_newa.append(X_outputadd_layer[j][i])
            X_newa.append(add_neg(X_outputadd_layer[j][i]))
        X_newa=np.array(X_newa)
        X_new_final.append(X_newa)
    return X_new_final


# def Added_data_layer(X_outputadd_layer):
#     X_newa=[]
#     for i in range(X_outputadd_layer.shape[0]):
#         X_newa.append(X_outputadd_layer[i])
#         X_newa.append(add_neg(X_outputadd_layer[i]))
#     X_newa=np.array(X_newa)
#     return X_newa



def Added_data_layer(X_outputadd_layer):
    X_newa = []
    for i in range(X_outputadd_layer.shape[0]):
        X_newa.append(X_outputadd_layer[i])
        X_newa.append(add_neg(X_outputadd_layer[i]))
    # Stack the list into a PyTorch tensor and ensure it is transferred to GPU if not already
    X_newa = torch.stack(X_newa)
    return X_newa

def pca_feature_layer_change(X_output_layer,n):
    X_pca_try=[]
    for i in range(len(X_output_layer)):
        X_pca_layers = sp.PCA_scratch(X_output_layer[i], n)
        # normalize the pca components 
        pca_components_norm=normalize(X_pca_layers,axis=1)
        #project the zerocentered data on the normalized dpca components 
        X_pca_try.append(np.dot(np.dot(X_output_layer[i],np.transpose(pca_components_norm)),pca_components_norm))
    return X_pca_try

def pca_change(X_output_layer,X_pca_try):
    X_pca=[]
    for i in range(len(X_output_layer)):
        #project the zerocentered data on the normalized dpca components 
        X_pca.append(np.dot(np.dot(X_output_layer[i],np.transpose(X_pca_try[i])),X_pca_try[i]))
    return X_pca


# def pca_change_layer(X_output_layer,X_pca_try):
#     X_pca=np.dot(np.dot(X_output_layer,np.transpose(X_pca_try)),X_pca_try)
#     return X_pca

def pca_change_layer(X_output_layer, X_pca_try):
    X_pca = torch.matmul(torch.matmul(X_output_layer, X_pca_try.T), X_pca_try)
    return X_pca

