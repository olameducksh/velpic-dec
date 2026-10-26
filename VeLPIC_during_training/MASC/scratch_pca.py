### pca_from_scratch
# zero centered data will be sent to this function
from sklearn.preprocessing import normalize
import numpy as np

def PCA_components(sorted_eigenvalue , sorted_eigenvectors, num_components):
      
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    eigenvalue_subset = np.sum(sorted_eigenvalue[0:num_components])/np.sum(sorted_eigenvalue)
    return eigenvector_subset.transpose(),eigenvalue_subset

def PCA_percent(sorted_eigenvalue , sorted_eigenvectors, n):
    # Determine explained variance
    total_egnvalues = sum(sorted_eigenvalue)
    var_exp = [(i/total_egnvalues) for i in sorted_eigenvalue]

    c_var_exp=np.cumsum(var_exp)
    num_components=0
    for i in range(len(c_var_exp)):
        if c_var_exp[i]<=n:
            num_components=num_components+1
    num_components=num_components+1        
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]

    return eigenvector_subset.transpose()

def PCA_function(X_meaned):
#     X_meaned = X - np.mean(X , axis = 0)

    cov_mat = np.cov(X_meaned , rowvar = False)
    
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    
    return sorted_eigenvalue , sorted_eigenvectors



def PCA_scratch(X,n):
    sorted_eigenvalue , sorted_eigenvectors=PCA_function(X)
    if n>=1:
        eigenvector_subset,eigenvalue_subset=PCA_components(sorted_eigenvalue,
                                          sorted_eigenvectors, n)
        return eigenvector_subset,eigenvalue_subset
    else:
        eigenvector_subset=PCA_percent(sorted_eigenvalue,
                                       sorted_eigenvectors, n)
        
        return eigenvector_subset
    

def PCA_new(X_output_layer,n):
    pca_components_norm=[]
    for i in range(len(X_output_layer)):
        X_pca_layers = PCA_scratch(X_output_layer[i], n)
        pca_components_norm.append(normalize(X_pca_layers,axis=1))
    return pca_components_norm

def PCA_new_layer(X_output_layer,n):
    if n>=1: 
        X_pca_layers,eigenvalue_subset = PCA_scratch(X_output_layer, n)
        pca_components_norm=normalize(X_pca_layers,axis=1)
        return pca_components_norm,eigenvalue_subset
    else:
        X_pca_layers = PCA_scratch(X_output_layer, n)
        pca_components_norm=normalize(X_pca_layers,axis=1)
        return pca_components_norm

