from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
NOISE_DIM = 96

def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.
    
    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
        random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    
    noise = torch.rand([batch_size, noise_dim], dtype=dtype, device=device)
    noise = -2 * noise + 1.0

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code

    INPUT_DIMS = 784
    HIDDEN_DIMS = 256
    N_CLASSES = 1
    
    model = nn.Sequential(
        nn.Flatten(),

        nn.Linear(in_features=INPUT_DIMS, out_features=HIDDEN_DIMS),
        nn.LeakyReLU(negative_slope=0.01),

        nn.Linear(in_features=HIDDEN_DIMS, out_features=HIDDEN_DIMS),
        nn.LeakyReLU(negative_slope=0.01),

        nn.Linear(in_features=HIDDEN_DIMS, out_features=N_CLASSES),
    )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code

    HIDDEN_DIMS = 1024
    IMAGE_DIMS = 784
    
    model = nn.Sequential(

        nn.Linear(in_features=noise_dim, out_features=HIDDEN_DIMS),
        nn.ReLU(),

        nn.Linear(in_features=HIDDEN_DIMS, out_features=HIDDEN_DIMS),
        nn.ReLU(),

        nn.Linear(in_features=HIDDEN_DIMS, out_features=IMAGE_DIMS),
        nn.Tanh()
    )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model  


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    
    labels_real = torch.ones_like(logits_real, dtype=logits_real.dtype, 
        device=logits_real.device)
    loss_real = nn.functional.binary_cross_entropy_with_logits(input=logits_real, 
        target=labels_real, reduction='mean')

    labels_fake = torch.zeros_like(logits_fake, dtype=logits_fake.dtype, 
        device=logits_fake.device)
    loss_fake = nn.functional.binary_cross_entropy_with_logits(input=logits_fake, 
        target=labels_fake, reduction='mean')

    loss = loss_real + loss_fake 

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    
    # the trick here is to set y = 1 in the formula:
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # logits_fake = D(G(z))
    labels_fake = torch.ones_like(logits_fake, dtype=logits_fake.dtype, 
        device=logits_fake.device)
    loss = nn.functional.binary_cross_entropy_with_logits(input=logits_fake, 
        target=labels_fake, reduction='mean')

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    
    # beta1 = .5 is NOT default (.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    # Replace "pass" statement with your code

    loss = (scores_real - 1) ** 2 + (scores_fake) ** 2 
    loss = torch.mean(loss)
    loss *= .5

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    # Replace "pass" statement with your code
    
    loss = (scores_fake - 1) ** 2
    loss = torch.mean(loss)
    loss *= .5

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
    the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    
    model = nn.Sequential(

        # (N, 784) -> (N, 1, 28, 28)
        # in pytorch we have to keep channels before image size
        nn.Unflatten(1, (1, 28, 28)),

        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1),
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(kernel_size=(2, 2), stride=2),

        # (N, 64, 4, 4) -> (N, 64*4*4)
        # flatten to 2D tensor: preserve batch size and flatten image dims into 
        # a single vector of size 64*4*4
        nn.Flatten(start_dim=1),

        # we have to compute in_features
        nn.Linear(in_features=4*4*64, out_features=4*4*64),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(in_features=4*4*64, out_features=1)

    )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
    the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    
    model = nn.Sequential(

        nn.Linear(in_features=noise_dim, out_features=1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),

        # (N, 1024) -> (N, 7*7*128)
        nn.Linear(in_features=1024, out_features=7*7*128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=7*7*128),

        # (N, 7*7*128) -> (N, 128, 7, 7)
        nn.Unflatten(1, (128, 7, 7)),

        # (N, 128, 7, 7) -> (N, 64, 14, 14) 
        # see formula here (it's just an inverse for a regular formula):
        # https://blog.paperspace.com/transpose-convolution/
        nn.ConvTranspose2d(in_channels=128, out_channels=64, 
            kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=64),

        # (N, 64, 4, 4) -> (N, 1, 28, 28)
        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4,
                      stride=2, padding=1),
        nn.Tanh(),

        nn.Flatten()
    )

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
