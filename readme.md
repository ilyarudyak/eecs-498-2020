# assignment 4
## Q1 - pytorch autograd

### barebones python
To train a barebones model we can use raw tensor operations or thin wrappers:
- initialize parameters using `nn.init.kaiming_normal_`; we also have manually require 
gradients on our weights using `.requires_grad_()`;
- write a forward pass as a function using `nn.functional` `F.linear` and `F.relu`; 
here `F.linear(x, w, b)` is just a wrapper for `x @ w.T + b`; and `F.relu` is a wrapper
for a strange function `F.clamp(min=0)`;

### module API
- instead of initializing our parameters manually we use classes from `nn.Module` like 
`nn.Conv2d`; tensors are created with `required_grad = True` by default;
- instead of creating a function for forward pass we create a class (which subclass `nn.Module`);
we initialize layers in the constructor and create the net connectivity in `forward` method; 
- its always a bit confusing that instead of calling `forward` directly we call the class itself:
`out = model(x)`;

## Q2 - image captioning
## Q3 - network visualization

All 3 questions are using a backward pass. Forward pass is used only for illustration purposes in lectures. But we use it in a different way: number of passes can be different as well as the initial image ans so on:

| question             | initial image | ## of passes | loss                  | regularization | 
| -------------------- | ------------- | ------------ | --------------------- | -------------- |
| saliency maps        | actual image  | single       | cross-entropy         | none           | 
| adversarial attack   | actual image  | multiple     | score of target class | none           |
| class visualization  | empty image   | multiple     | correct score         | l2             |

In all cases we use  gradient *ascent* which means we *increase* the value of a variable by a gradient (not *decrease* it as usual). We should use an appropriate scope and remove accumulated gradients as specified in this [tutorial](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html).

### 01 - saliency maps
- there's no optimization problem - we just run a single pass of a gradient ascent (which makes it pretty quick); why does it work?

> In this case (a linear model), it is easy to see that the magnitude of elements of `w` defines the importance of the corresponding pixels of `I` for the class `c`. ... given an image `I0`, we can approximate `Sc(I)` with a linear function in the neighbourhood of `I0` by computing the first-order Taylor expansion.

> Another interpretation of computing the image-specific class saliency using the class score derivative (4) is that the magnitude of the derivative indicates which pixels need to be changed the least to affect the class score the most.

- it's not quite clear from the paper what should we backpropagate - class scores or cross-entropy loss; it seems that cross-entropy loss will give us pretty good saliency maps;

### 02 - adversarial attack
- it's quite easy to fool a model and it's not stable at all:

> For all the networks we studied (MNIST, QuocNet [10], AlexNet [9]), for each sample, we have always managed to generate very close, visually hard to distinguish, ad- versarial examples that are misclassified by the original network.

- the **optimization problem**: `max Sc(I)` over `I` where `c` is a target class; so we run gradient ascent trying to maximize the target (adversarial) class on the provided image; 
- we use an actual image (for example an image of a schoolbus) and construct and adversarial image; it visually looks like exactly the same but the model gives us the different (adversarial) class (for example an ostrich);
- we run the backward pass until the model is fooled; the number of iterations is usually quite small (10-20); it's better to run a few iterations (2-3) after the target class is reached to get a stable result;
- we should use normalized gradient;

### 03 - class visualization
- the main idea - we may reconstruct an image (which is going to look like a deep dream image) that the model classifies as a given class:

> The procedure is related to the ConvNet training procedure, where the back-propagation is used to optimise the layer weights. The difference is that in our case the optimisation is performed with respect to the input image, while the weights are fixed to those found during the training stage.

- the **optimization problem**: `max Sc(I) + l2` over `I` starting from an empty image; 
- we backpropagate from a class score rather than from softmax probability:

> The reason is that the maximization of the class posterior can be achieved by minimising the scores of other classes. Therefore, we optimise `Sc` to ensure that the optimisation concentrates only on the class in question `c`. 

## Q4 - style transfer
There are 3 main ideas behind style transfer: 
- first of all we may construct an image using backprop like before; 

> To visualise the image information that is encoded at
different layers of the hierarchy one can perform gradient descent on a white noise image to find another image that matches the feature responses of the original image (Fig 1, content reconstructions) [24].

- the key idea - we may extract content and style from an image *separately*: to extract content we may use feature maps from upper convolutional layers; to extract style we have to use a covariance matrix of features (Gram matrix); 

> We therefore refer to the feature responses in higher layers of the network as the content representation. To obtain a representation of the style of an input image, we use a feature space designed to capture texture information [10].


- so we extract content from one image and style from another; we have 2 loss functions for backprop and we may combine them (in the paper they combine them with some coefficients, in the assignment we just use sum of them);

> Thus we jointly minimise the distance of the feature representations of a white noise image from the content representation of the photograph in one layer and the style representation of the painting defined on a number of layers of the Convolutional Neural Network.

# assignment 5. object detection.
## Q1 YOLO

### `ReferenceOnActivatedAnchors`
- This is written for us but it's better to understand what this function is doing. The purpose of the fuction is more or less clear but its vectorized implementation is quite involved. Here's a brief analysis of the function (only for the case of `YOLO`). 
- Much more detailed analysis is in a separate debugging file.
- We illustrate computations with the image of a car (image 0 in the notebook).

#### The purpose of the function:
- We compute *activated* anchor boxes:
	- We have the following items: 1) an image; 2) a GT bbox (just one for simplicity); 3) grid centers. 
	- We do the following steps: 1) identify *responsible* cell - the center of GT bbox should be in this cell (cell `[3, 3]` in our case); 2) consider anchor boxes only from this cell; compute IoU between them and GT bbox (A=9 numbers); choose the anchor boxes with max IoU (`[0.1007, 0.4029, 0.7454, 0.5733, 0.3971, 0.5918, 0.5216, 0.6619, 0.4799]` in our example; max is `.7454` - so we need the anchor box with `index=2`). So in our example activated anchor box for our example is `[2, 3, 3]`.

#### How do we compute it?
- We have to compute 3 masks: 1) `bbox_mask` (B, N); 2) `grid_mask` (B, 1, H' * W', N); 3) `anc_mask` (B, A, H' * W', N):
	- `grid_mask`: (a) it shows us a *responsible* cell (again the center of GT bbox is in this cell) for each GT bbox; (b) we don't locate somehow the center of a GT bbox, rather we just take the closest cell (min distance between centers of the cell and GT bbox; we use manhattan distance); (c) in our case this is cell `[3, 3]` (which corresponds to index 24 if we reshape `7x7` to `49`); in fact `grid_mask[0, 0, :, 0]` contains only one True value at index 24; (d) why is it this cell? well it has coordinates `[250.0000, 166.5000]`; GT bbox center is `[253.5000, 183.5000]` and the manhattan distance is 20.5 (and next grid center has distance 34.0714); in the debugging file we have a picture that shows that this is in fact the closest grid center;
	- `anc_mask`: if we look at the shape of this mask we may see that 2 parameters are fixed; we fixed GT bbox and *responsible* grid cell for it; we now consider all `A=9` anchor boxes only for this grid cell; how can we define *activated* anchor box? we just compute IoU between those 9 anchor boxes and GT box; in our example its anchor with `index=2`; so we have `[2, 3, 3]` *activated* anchor box which is 122 in reshaped to `2*7*7` tensor; that's exactly what we get from this function: `activated_anc_ind = [122, 605, 871, 955]`.

### `PredictionNetwork`
- This is a very light network - only 2 layers of `1x1` convolution (as we remember the majority of the work is performed by the backbone net). We need to get quite a lot of output (for each cell in `7x7` grid): 1) `C` classes; 2) `4A` offsets for anchor boxes; 3) `A` confidence scores. So out has the shape `(B, 5*A+C, H, W)`. We assume that cofidence scores is at `(:, :A, :, :)`.
- The only complication in this fuction - during training we have to use only activated anchors. THis is technically less involved as the main complications are inside the previous function . 

## Q2 Faster R-CNN














