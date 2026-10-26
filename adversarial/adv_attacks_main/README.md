# Pytorch Adversarial Attacks

Custom Adversarial Attacks implemented in PyTorch.

### Attack Support
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
### Dataset Support
- MNIST
- CIFAR10
  
### Usage
#### `/attacks.py`
This file has the implemented adversarial attacks. You can call each attack by itself like below or use the `eval_attack` function that returns the adversarial accuracy for a specified attack and test dataset.

- FGSM Attack
```
fgsm_attack(model, x, y, attack_params): 
    # args
    # model: nn.Module - model to be attacked
    # x: torch.Tensor - batched image input
    # attack_params: dict
    #       eps: float - epsilon budget
    #       targetted: bool - whether to use targeted attack
    # 
    # returns: torch.Tensor - perturbed image (batched)
```
- PGD Attack
```
pgd_attack(model, x, y, attack_params):
    # args
    # model: nn.Module - model to be attacked
        # x: torch.Tensor - batched image input
        # y: torch.Tensor - batched ground truth labels (or target labels if targeted is True)
        # attack_params: dict
                # eps: float - epsilon budget
                # targeted: bool - whether to use targeted attack
                # iters: int - number of attack iterations
                # step_size: float - step size per iteration
```
#### `/benchmark.py`
This file has code to visualise example perturbations and plot the epsilon vs accuracy plots. Example usage of these attacks can be seen here.
