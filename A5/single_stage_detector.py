import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid, device='cuda'):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  
  # unpack shapes
  A, _ = anc.shape
  B, Hp, Wp, _ = grid.shape

  # initialize anchors
  #   assert anc.dtype == grid.dtype, 'dtype mismatch!'
  #   assert anc.device == grid.device, 'device mismatch!'
  dtype, device = anc.dtype, anc.device
  anchors = torch.zeros((B, A, Hp, Wp, 4), device=device)

  # build anchors
  for b in range(B):
    for a in range(A):
      for i in range(Hp):
        for j in range(Wp):

          # this order seems to work for the test case
          x, y = grid[b, i, j]
          # this order is explicitely provided above
          w, h = anc[a]

          # w corresponds to x axis, h - to y axis
          x_tl, y_tl = x - .5 * w, y - .5 * h
          x_br, y_br = x + .5 * w, y + .5 * h
          anchors[b, a, i, j, :] = torch.Tensor([x_tl, y_tl, x_br, y_br])

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO', device='cuda'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])

  anchors, offsets = anchors.to(device), offsets.to(device)

  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code

  # initialize proposals
  B, A, Hp, Wp, _ = anchors.shape
  # assert anchors.dtype == offsets.dtype
  # assert anchors.device == offsets.device
  # dtype = anchors.dtype
  # device = anchors.device
  proposals = torch.zeros((B, A, Hp, Wp, 4), device=device)

  # build proposals
  for b in range(B):
    for a in range(A):
      for i in range(Hp):
        for j in range(Wp):

            # (1) transform (x_tl, y_tl, x_br, y_br) -> (xc, yc, w, h)
            x_tl, y_tl, x_br, y_br = anchors[b, a, i, j, :]
            w, h = x_br - x_tl, y_br - y_tl
            xc, yc = x_tl + .5 * w, y_tl + .5 * h 

            # (2) apply transformation with (tx, ty, tw, th)
            tx, ty, tw, th = offsets[b, a, i, j, :]
            if method == 'YOLO':
              xc += tx
              yc += ty 
            else: # method == 'FasterRCNN'
              xc += tx * w
              yc += ty * h

            # for both methods
            w *= torch.exp(tw)
            h *= torch.exp(th)

            # (3) transform back (xc, yc, w, h) -> (x_tl, y_tl, x_br, y_br)
            x_tl, y_tl = xc - .5 * w, yc - .5 * h
            x_br, y_br = xc + .5 * w, yc + .5 * h
            proposals[b, a, i, j, :] = torch.Tensor([x_tl, y_tl, x_br, y_br]).to(device)           


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals.to(device)


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None

  proposals = proposals.to('cuda')
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  
  # unpack shapes
  B, A, Hp, Wp, _ = proposals.shape
  _, N, _ = bboxes.shape 

  # repeat proposals and bboxes
  proposals_resh = proposals.reshape(B, A*Hp*Wp, 4).unsqueeze(dim=2)
  ps = proposals_resh.repeat(1, 1, N, 1)
  # we can achieve the same effect with broadcasting but 
  # it's better to keep everything as clear as possible
  bb = bboxes[:, :, :4].unsqueeze(dim=1).repeat(1, A*Hp*Wp, 1, 1)

  # compute simple areas; we use the fact that x_br >= x_tl and the same for y
  # (B, A*Hp*Wp, N, 4)
  dx = ps[..., 2] - ps[..., 0]
  dy = ps[..., 3] - ps[..., 1]
  area_proposal = dx * dy

  # (B, A*Hp*Wp, N, 4)
  dx = bb[..., 2] - bb[..., 0]
  dy = bb[..., 3] - bb[..., 1]
  area_bboxes = dx * dy

  # compute intersection area using min / max
  # https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
  # if we have a negative number this means we don't have intersection, so we have 
  # to eliminate those cases with torch.clamp()
  xp_tl, yp_tl, xp_br, yp_br = ps[..., 0], ps[..., 1], ps[..., 2], ps[..., 3]
  xb_tl, yb_tl, xb_br, yb_br = bb[..., 0], bb[..., 1], bb[..., 2], bb[..., 3]

  dx = torch.min(xb_br, xp_br) - torch.max(xb_tl, xp_tl)
  dx = torch.clamp(dx, min=0)

  dy = torch.min(yb_br, yp_br) - torch.max(yb_tl, yp_tl)
  dy = torch.clamp(dy, min=0)

  area_inters = dx * dy

  # compute iou_mat
  iou_mat = area_inters / (area_proposal + area_bboxes - area_inters)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.

    out_dim = int(5 * self.num_anchors + self.num_classes)
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 1)),
      nn.Dropout(p=drop_ratio),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=(1, 1))
    )

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    
    # forward pass
    # (B, 5 * A + C, H, W)
    out = self.pred_layer(features)
    B, _, H, W = out.shape

    # if pos_anchor_idx is provided - training
    if not pos_anchor_idx is None:

      ########################## conf_score and offsets ##########################

      # (B, A, 5, 7, 7) -> (M, 5)
      out_anchor_data = out[:, :self.num_anchors*5, :, :]
      out_anchor_data_resh = out_anchor_data.reshape(B, self.num_anchors, 5, H, W)
      pos_ancor_data = self._extract_anchor_data(out_anchor_data_resh, pos_anchor_idx)
      neg_ancor_data = self._extract_anchor_data(out_anchor_data_resh, neg_anchor_idx)

      # we assume that conf_score has index 0
      # conf_scores required shape: (2 * M, 1)
      M, _ = pos_ancor_data.shape
      pos_conf_score = pos_ancor_data[:, 0].reshape(M, 1) # (M, 1)
      neg_conf_score = neg_ancor_data[:, 0].reshape(M, 1) # (M, 1)
      conf_scores = torch.vstack([pos_conf_score, neg_conf_score]) # (2 * M, 1)

      # make sure conf_scores is between 0 and 1 
      conf_scores = torch.sigmoid(conf_scores)

      # offsets required shape (only for positive anchors): (M, 4)
      offsets = pos_ancor_data[:, 1:].clone()

      # make sure the first two elements t^x and t^y of offsets are between -0.5 and 0.5
      offsets[:, :2] = torch.sigmoid(offsets[:, :2]) - .5

      ############################### class scores ###############################

      # class_scores required shape (only for positive anchors): (M, C)
      # (B, C, 7, 7) -> (M, C)
      out_class_score = out[:, self.num_anchors*5:, :, :] 
      class_scores = self._extract_class_scores(out_class_score, pos_anchor_idx)

    # pos_anchor_idx is NOT provided - testing 
    else:
      
      # conf_scores required shape: (B, A, H, W)
      conf_scores = out[:, 0:self.num_anchors, :, :]
      conf_scores = torch.sigmoid(conf_scores)

      # offsets required shape: (B, A, 4, H, W)
      offsets = out[:, self.num_anchors:5*self.num_anchors, :, :].clone()
      offsets = offsets.reshape(B, self.num_anchors, 4, H, W)
      offsets[:, :, :2, :, :] = torch.sigmoid(offsets[:, :, :2, :, :]) - .5

      # class_scores: (B, C, H, W)
      class_scores = out[: , 5*self.num_anchors:, :, :]


    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):

  def __init__(self, device='cuda', method='YOLO'):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    # no pooling, so output (BS, 1280, 7, 7)
    self.feat_extractor = FeatureExtractor() 
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
    self.device = device
    self.method = method

  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    
    # (i) extract image features: (B, 3, 224, 224) -> (B, 1280, 7, 7)
    features = self.feat_extractor(images)

    # (ii) generate grid and anchors
    grid_centers = GenerateGrid(batch_size=images.shape[0], device=self.device) # (B, 7, 7, 2)
    anchors = GenerateAnchor(self.anchor_list, grid_centers) # (B, A, 7, 7, 4)

    # (iii) IoU between anchors and GT boxes
    # (B, A, 7, 7, 4), (B, N, 5) -> (B, A*7*7, N)
    # we use here anchors, not proposals
    iou_mat = IoU(proposals=anchors, bboxes=bboxes)
    # it's time to use our involved function to get activated anchors
    activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, _, _ = \
    ReferenceOnActivatedAnchors(
      anchors=anchors, bboxes=bboxes, grid=grid_centers, 
      iou_mat=iou_mat, pos_thresh=0.7, 
      # changed from default .3 only for YOLO (see HINT above)
      neg_thresh=0.2, method=self.method)

    # (iv) conf_scores, offsets, class_prob through the prediction network
    conf_scores, offsets, class_scores = self.pred_network(
      features=features, pos_anchor_idx=activated_anc_ind, neg_anchor_idx=negative_anc_ind)

    # (v) compute loss
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)

    anc_per_img =  anchors.shape[1] * anchors.shape[2] * anchors.shape[3] # A * 7 * 7
    cls_loss = ObjectClassification(class_scores, GT_class, images.shape[0], 
      anc_per_img, activated_anc_ind)

    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code
    out = self.feat_extractor(images)
    N = images.shape[0]
    grid = GenerateGrid(N)
    anchors = GenerateAnchor(self.anchor_list, grid)
    # iou_mat =  IoU(anchors, bboxes)
    # activated_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class, activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat,neg_thresh=0.2)
    conf_scores, offsets, class_scores =  self.pred_network(out)
    # print(conf_scores.shape)
    # print(offsets.shape)
    # print(class_scores.shape)
    offsets = offsets.permute(0,1,3,4,2)
    # print('permute(0,1,3,4,2)',offsets.shape)
    offsets = GenerateProposal(anchors, offsets, method='YOLO')
    # print('GenerateProposal',offsets.shape)
    offsets = offsets.permute(0,1,4,2,3).double()
    # print('permute(0,1,4,2,3)',offsets.shape)
    conf_scores, offsets, class_scores =  conf_scores.reshape(conf_scores.shape[0],conf_scores.shape[1],-1), offsets.reshape(offsets.shape[0],offsets.shape[1],offsets.shape[2],-1), class_scores.reshape(class_scores.shape[0],class_scores.shape[1],-1)
    offsets = offsets.permute(0,1,3,2)
   

    conf_scores = conf_scores.permute(0,2,1)
    offsets = offsets.permute(0,2,1,3)
    class_scores =class_scores.permute(0,2,1)
    # print(conf_scores.shape)
    # print(offsets.shape)
    # print(class_scores.shape)
    for n in range(N):
      local_conf_scores = conf_scores[n]
      local_offsets =  offsets[n]
      local_class_scores,idx_class_scores = class_scores[n].max(dim = 1)
      # print(local_conf_scores.shape)
      # print(local_offsets.shape)
      # print(idx_class_scores.shape)
      target = torch.zeros(local_offsets.shape[0],local_offsets.shape[1],local_offsets.shape[2]+2)
      target[:,:,:4] = local_offsets
      target[:,:,4] = local_conf_scores
      # print(target[:,:,6].shape)
      # print(idx_class_scores.unsqueeze(1).repeat(1,9).shape)
      target[:,:,5] = idx_class_scores.unsqueeze(1).repeat(1,9)
      target = target.reshape(-1,6)
      target = target[target[:,4]>thresh]
      idx = torchvision.ops.nms(target[:,:4], target[:,4].reshape(-1), nms_thresh)
      target = target[idx]
      final_proposals.append(target[:,:4].detach())
      final_conf_scores.append(target[:,4].reshape(-1,1).detach())
      final_class.append(target[:,5].reshape(-1,1).detach())
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  print(f'iou_threshold:{iou_threshold}')

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  
  def get_iou_mat(boxes):
    proposals = boxes.reshape(1, boxes.shape[0], 1, 1, 4)
    bboxes = boxes.reshape(1, boxes.shape[0], 4)
    return IoU(proposals, bboxes).squeeze(0)

  def sort_boxes(boxes, scores):
    _, idxs_sorted = torch.sort(scores, descending=True)
    boxes_sorted = boxes[idxs_sorted]
    return boxes_sorted, idxs_sorted

  ####################################################################  

  # mask element
  MASK = -1
  
  # get sorted boxes and idxs
  # get iou_mat for sorted boxes
  boxes_sorted, idxs_sorted = sort_boxes(boxes, scores)
  iou_mat_sorted = get_iou_mat(boxes_sorted)
  idxs_sorted_loop = idxs_sorted.clone()
  
  # we don't delete any boxes 
  # instead we mark idxs in idxs_sorted as -1
  # it can be changed back
  # so we iterate while we have non -1 idxs in idxs_sorted
  keep_idxs = []
  while torch.any(idxs_sorted_loop != MASK):
  
      # take the first idx in idxs_sorted that's not -1 (max_idx)
      # get original idx for it: idxs_sorted[max_idx]
      # add this original idx to keep_idxs
      max_idx = torch.nonzero(idxs_sorted_loop != MASK)[0].item()
      orig_idx = idxs_sorted[max_idx]
      keep_idxs.append(orig_idx)
      
      # get the row of iou_mat for this max_idx
      # compare the row with iou_threshold
      # mark idxs in idxs_sorted as -1
      # and that's it
      idxs_to_discard = iou_mat_sorted[max_idx] > iou_threshold
      idxs_sorted_loop[idxs_to_discard] = MASK
      
  keep = torch.Tensor(keep_idxs)
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep.long()


def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

