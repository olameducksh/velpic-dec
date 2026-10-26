import torch
#extracting features from each layer for the data
def classwise_data(X, y, number_class=10):
    # Dictionary holding the classes dataset
    obj = {f'class{i}': [] for i in range(number_class)}

    for i in range(len(y)):
        obj[f'class{y[i]}'].append(X[i])

    # Convert lists to tensors
    for key in obj:
        obj[key] = torch.stack(obj[key]) if obj[key] else torch.empty((0, X.size(1)))  # Create an empty tensor if the class has no samples

    return obj

# add data code
def add_neg(arr):
    return (-1 * arr)

def Added_data_layer(X_outputadd_layer):
    X_newa = []
    for i in range(X_outputadd_layer.shape[0]):
        X_newa.append(X_outputadd_layer[i])
        X_newa.append(add_neg(X_outputadd_layer[i]))
    # Stack the list into a PyTorch tensor and ensure it is transferred to GPU if not already
    X_newa = torch.stack(X_newa)
    return X_newa

def pca_change_layer(X_output_layer, X_pca_try):
    X_pca = torch.matmul(torch.matmul(X_output_layer, X_pca_try.T), X_pca_try)
    return X_pca