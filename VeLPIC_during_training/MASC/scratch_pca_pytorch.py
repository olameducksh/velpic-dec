import torch
import torch.nn.functional as F  # For normalization

# Function to get PCA components
def PCA_components(sorted_eigenvalue, sorted_eigenvectors, num_components):
    # Select the top 'num_components' eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, :num_components]
    return eigenvector_subset.T

# Function to select components based on explained variance percentage
def PCA_percent(sorted_eigenvalue, sorted_eigenvectors, n):
    total_eigenvalues = torch.sum(sorted_eigenvalue)
    var_exp = sorted_eigenvalue / total_eigenvalues

    # Cumulative sum of explained variance
    c_var_exp = torch.cumsum(var_exp, dim=0)

    num_components = torch.sum(c_var_exp <= n).item() + 1
    eigenvector_subset = sorted_eigenvectors[:, :num_components]

    return eigenvector_subset.T

# Function to calculate eigenvalues and eigenvectors for the covariance matrix
def PCA_function(X_meaned):
    # Covariance matrix
    cov_mat = torch.cov(X_meaned.T)
    
    # Get eigenvalues and eigenvectors
    eigen_values, eigen_vectors = torch.linalg.eigh(cov_mat)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_index = torch.argsort(eigen_values, descending=True)
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    
    return sorted_eigenvalue, sorted_eigenvectors

# Wrapper function for PCA calculation based on component number or variance percentage
def PCA_scratch(X, n):
    sorted_eigenvalue, sorted_eigenvectors = PCA_function(X)
    if n >= 1:
        eigenvector_subset = PCA_components(sorted_eigenvalue, sorted_eigenvectors, n)
    else:
        eigenvector_subset = PCA_percent(sorted_eigenvalue, sorted_eigenvectors, n)
    return eigenvector_subset

# Apply PCA and normalize for a list of data
def PCA_new(X_output_layer, n):
    pca_components_norm = []
    for i in range(len(X_output_layer)):
        X_pca_layers = PCA_scratch(X_output_layer[i], n)
        # Normalize using PyTorch's F.normalize
        pca_components_norm.append(F.normalize(X_pca_layers, p=2, dim=1))
    return pca_components_norm

# Apply PCA and normalize for a single data layer
def PCA_new_layer(X_output_layer, n):
    X_pca_layers = PCA_scratch(X_output_layer, n)
    # Normalize using PyTorch's F.normalize
    pca_components_norm = F.normalize(X_pca_layers, p=2, dim=1)
    return pca_components_norm
