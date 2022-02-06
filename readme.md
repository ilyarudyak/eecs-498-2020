# assignment 3
This is a pretty big assignment. Without prior experience with cs231n it would probably take 2-3 weeks (or more) to do it properly from scratch. The main issue is of course to get backpropagation right but there are many other issues. For example in `SpatialBatchNorm` you should be very careful with permuting dimensions and so on.

## Q1 - fully connected networks

- This is more or less straightforward question. The most nontrivial question is as usual - backpropagation. But in this case we have a detailed description in cs231n notes [link](https://cs231n.github.io/linear-classify/) and [link](https://cs231n.github.io/neural-networks-case-study/).

## Q2 - convolutional networks

### overfit small data

- It turns out that without proper initialization and using batch norm it's really hard to get a correct combination of learning rate and weight scale. As we may see in this example basically all combinations give us `train_acc=.160000` after 10 epochs. This means the net just doesn't learn anything.
- After implementing Kaiming initialization we may see that even within 1 epoch it helps the net to get accuracy almost 2 times higher.

### Kaiming initialization
- There's no question about why we should do this (from *He et al., 2015*):

> This product is the key to the initialization design. A proper initialization method should avoid reducing or magnifying the magnitudes of input signals exponentially.

- There's also no question what gain we should use: formulas (7) and (8), (9) are different precisely because we use ReLU. And we can infer gain of 2 from the (9) formula.

- The main question here - what formula should we actually use? In *Glorot et al.* we may read about formula that uses both `n_in` and `n_out`. In the second paper *He et al.* we may see `n_l` without clear specification what is that. 
- Fortunately the correct formulas are specified in the assignment: 1) in case of a linear net we use `n_in`; 2) in case of a convolution net we also use `n_in` but we have to compute it: `num_in_channels * K * K `, where `K` is the number of a filter.

### train a good model

- We trained a model with 75.18% accuracy on the validation set and 74.68% on the test set. That's higher than mentioned in the notebook. 
- Within 60 sec. we don't have too much flexibility in building a net. So we just take the first half of VGG 11 and modified it a little bit. We don't modify the learning rate - everything works with the rate that was used in the notebook in previous exercises.

```python
...
num_filters=[32, 64, 128, 128],
max_pools=[0, 1, 3]
...
```

### batch norm
#### forward pass
For the training pass we have to implement `Algorithm 1` from *Ioffe et al., 2015* (p.3):

- compute mean and var over mini-batch;
- normalize x based on mean and var of mini-batch;
- scale and shift normalized x based on gamma and beta;

Why do we need to scale and shift?

> Note that simply normalizing each input of a layer may change what the layer can represent. For instance, nor- malizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. To address this, we make sure that the transformation inserted in the network can represent the identity transform.

We also keep tracking of running mean and var and use them during test pass. We use them instead of computing mean and variance over test mini-batch (this is not mentioned in the paper). 

#### backward pass
We implemented 2 approaches:

- Computation graph (it's rather tricky and requires a separate post);
- Using analytical form of the gradient. A good place to start - formulas in the paper. We can implement them as a first step. We may also simplify them - there are at least 2 possible formulas (both of them are implemented). Here's my [gist](https://gist.github.com/ilyarudyak/55ff4d9c705964eb3dc83bde091d97a8) with deriving those formulas.

#### potential bug
I was not able to run code on `cuda` with batch normalization, I got the following error message (class `FastConv`):

> `RuntimeError: set_sizes_and_strides is not allowed on a Tensor created from .data or .detach().
If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset) without autograd tracking the change, remove the .data / .detach() call and wrap the change in a with torch.no_grad(): block`.

But it seems we can wrap it in this block and it works (see details in the file). 

#### applications
We have 2 toy examples in the assignment that illustrate usefullness of a batch norm. 

- First we train on a small sett of 500 examples with some fixed `lr=1e-3`: we may see that the net with BN has much better training accuracy (.95 vs .43) within 10 epochs. This is a hint that a net without BN is much more sensitive to a learning rate.
- We check this in the second example. We may see that training accuracy: 1) less depends on learning rate; and 2) it's a bit higher than accuracy without BN in general.

#### spatial batch norm
- That's in fact can be done using our previous `BatchNorm` class. The only thing we should be careful with - we need to deal with dimensions carefully. What we need - average over N, H, W instead of N in a regular case. As explaned in A1 we can't just reshape - we need first switch dimensions:

```python
x2d = x.permute(0, 2, 3, 1).reshape(N*H*W, C)
```

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

### how to use markdown preview
- [markdown preview](https://github.com/facelessuser/MarkdownPreview)
- [key binding](https://plaintext-productivity.net/2-04-how-to-set-up-sublime-text-for-markdown-editing.html)












