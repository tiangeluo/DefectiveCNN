import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
import torch.nn.functional as F
import IPython

import torch
import torch.nn as nn

from utils import to_var

# --- White-box attacks ---

class FGSMAttack(object):
    def __init__(self, model=None, epsilon=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilons=None):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X

eps = 0.1
def label_smooth(pred, gold):
    gold = gold.contiguous().view(-1)
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    return torch.mean(loss)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
up = ((np.ones([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)
down = ((np.zeros([3,32,32]) - np.array(mean).reshape(3,1,1))/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)
bound_base = (np.ones([3,32,32])/np.array(std).reshape(3,1,1)).reshape(1,3,32,32)
stepsize = 8
learning_rate = (1./255/np.array(std)*stepsize).reshape(1, len(std), 1, 1)

def recover(kk):
    kk = kk[0,:,:,:]
    k1 = kk[0,:,:]
    k2 = kk[1,:,:]
    k3 = kk[2,:,:]
    k1 = (k1*std[0]+mean[0])*255
    k2 = (k2*std[1]+mean[1])*255
    k3 = (k3*std[2]+mean[2])*255
    return np.stack([k1,k2,k3])

class LinfPGDAttack(object):
    #def __init__(self, model=None, epsilon=4./255 * bound_base, k=6, a=learning_rate, 
    def __init__(self, model=None, epsilon=8./255 * bound_base, k=4, a=learning_rate, 
        random_start=False):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = label_smooth

    def perturb(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(torch.from_numpy(X).type(torch.FloatTensor), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)
            #IPython.embed()
            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, down, up) # ensure valid pixel range

        return X


# --- Black-box attacks ---

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        score = model(x_var)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])

    # For each input in the previous' substitute training iteration
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[len(X_sub_prev)+ind] = X_sub[ind] + lmbda * grad_val #???

    # Return augmented training data (needs to be labeled afterwards)
    return X_sub
