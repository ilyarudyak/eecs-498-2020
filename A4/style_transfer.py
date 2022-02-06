"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a4_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - scalar content loss
    """
    loss = None
    ##############################################################################
    # TODO: Compute the content loss for style transfer.                         #
    ##############################################################################
    # Replace "pass" statement with your code
    
    # unpack shapes
    _, C_l, H_l, W_l = content_current.shape

    # reshape activations
    M_l = H_l * W_l
    F = content_current.reshape(C_l, M_l)
    P = content_original.reshape(C_l, M_l)

    # compute loss
    loss = content_weight * torch.sum((F - P) ** 2)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return loss


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ##############################################################################
    # TODO: Compute the Gram matrix from features.                               #
    # Don't forget to implement for both normalized and non-normalized version   #
    ##############################################################################
    # Replace "pass" statement with your code
    
    # unpack the weghts
    N, C, H, W = features.shape

    # reshape features
    F = features.reshape(N, C, H * W)

    # compute gram matrix 
    # we may implement this using torch.bmm but
    # it looks like we use N = 1 in practice
    gram = torch.zeros((N, C, C), dtype=features.dtype, device=features.device)
    for n in range(N):
      gram[n] = F[n] @ F[n].T 

    # normalize matrix
    if normalize:
      gram /= (H * W * C)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    loss = None
    ##############################################################################
    # TODO: Computes the style loss at a set of layers.                          #
    # Hint: you can do this with one for loop over the style layers, and should  #
    # not be very much code (~5 lines).                                          #
    # You will need to use your gram_matrix function.                            #
    ##############################################################################
    # Replace "pass" statement with your code
    
    loss = 0
    # feats and style_targets have different size
    # so we may not use the same indexing
    for i, l in enumerate(style_layers):
      feat = feats[l]
      gram_feat = gram_matrix(feat)
      # looks like style_targets already contain gram matrices
      gram_target = style_targets[i]
      weight = style_weights[i]
      loss += weight * torch.sum((gram_feat - gram_target) ** 2)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    loss = None
    ##############################################################################
    # TODO: Compute total variation loss.                                        #
    # Your implementation should be vectorized and not require any loops!        #
    ##############################################################################
    # Replace "pass" statement with your code
    
    height_var = torch.sum((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2)
    width_var = torch.sum((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2)
    loss = tv_weight * (height_var + width_var)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return loss








