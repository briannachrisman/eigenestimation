
# Feature and Circuit Identification through Sparse Eigendecomposition


## Background

Mechanistic interpretability is somewhat of an emerging field in the deep learning space, and aims to bridge the gap between the seemingly black-box behavior of machine learning models and our understanding of their internal mechanisms. Rather than treating models as inscrutable input-output systems, mechanistic interpretability methods attempt to uncover the features, circuits, or higher-level abstractions that models have learned to represent. Being able to more precisely understand the internal computations and decisions a model is making, can potentially allow developers and users to identify biases or danger capabilities a model has (reference), debug poor performance during inference (references), and ideally intervene on a model's behavior post-training without the need for additional data (references).  


### Activation Space-Based Methods 

Most current mechanistic interpretability methodscan be categorized as supervised or unsupervised, although some are also a combination of the two. Noteably, almost all methods rely on extracing information from the **activation space** - the output vector of sets of neurons - of a model during inference time. 

**Supervised methods** generally rely on curated input-output pairs where a single "feature" - a human intepretable concept - has been changed between the two. For example, sets of pairs that isolate a "gender" feature in text might be pairs of sentences where the pronoun in "he/him/his" in the first of the pair and "she/her/hers" in the second, keeping everything else the same ("He went to the store / She went to the store", "His dog was his best friend/Her dog was her best friend"). To isolate a "sunny day" feature in a computer vision dataset, we might take pairs of photos of the same places on overcast and sunny days. From here, supervised methods attempt to identify a direction in activation space - a direction in residual stream space or the set of attention patterns at a specific layer, or perhaps the output of a convolutional layer in a computer-vision based model - that either most directly descends from the change in features between pairs, and that can most directly facilitate model behaviors related to the feature of interest.

Activation patching involves substituting activations from one model state (or input-output pair) into another during inference to observe whether specific features or behaviors transfer. For instance, patching an attention head's activations from a "female" input into a "male" input can help isolate whether that attention head is responsible for encoding gendered information ( )Similarly, attribution patching builds causal graphs to measure how much influence a particular neuron, layer, or attention head has on the downstream output, using targeted interventions to identify critical paths in the activation flow.  

**Unsupervised** mechanistic interpretability methods do not require such curated datasets or a particular feature of interest but instead focus on discovering latent structures in activation spaces. **Sparse autoencoders (SAEs)** are the foundation for these type of methods, and they attempt to decompose a chosen activation space into linear combination of vectors that theoretically represent feature directions. Relying on the assumption in such large multi-use case networks, a given input likely only activates a very small subset of features that the model has learned to represent. Imposing a sparsity constraint on the number of neurons that will be activated in an autoencoder with a given input, encourages the autoencoder to learn principal vectors of the activation space with minimal overlap that effectively isolate distinct concepts in separate activation dimensions.  Recently both unsupervised and supervised mechanistic interpretabilty have been combined with **causal graph methods** in order to build maps of a model's computation and pathways  and  better understand how high-level features combine and interact. (Geiger, Automated Circuit discovery)

Its worth noting that much of recent mechanistic interpretability research focuses on large language models (LLMs). Some of this is naturally because of the rising ubiquity of LLMs, but also because structure of language, and the transformer architecture that LLMs utilize, and the structure of language, lend themselves well to a few key assumptions criticial to many mechanistic interpretability research described above. First of all, for many features we are interested in on an application level (bias, refusal, syncophancy, refusal, misinformation) - with a bit of creativity, it is relatively straightforward to come up with sets of input pairs that can isolate such a feature. In many cases, with good prompt engineering language models can even help us curate such pairs. Secondly, autoregressive attention and the residual stream in the encoder-only transformer architectures popularly used for LLMs intuitively seem to promote features that can be **linearly represented**. Many studies have found numerous features, both simple and complex that are represented as directions from a single layer's MLP activation space, attention head output, or residual stream space - although there is also growing evidence that there may be non-linear features, feature manifolds, or features that split across multiple layers or blocks in a model that cannot be extracted with the current single-layer sparse autoencoder-based methods. 

### Limtations of activation-space based work 

This brings us to the limitations of activation-space based methods of mechanistic interpretability. To us, activation-spaced methods of mechanistic interpretability, have three major limitations. First of all, there is emerging evidence that even models with architectures that encourage linearity, such as a residual stream - may not have a linear latent spaces.  (Insert some references here). SAE-based methods, explicitly assume that the activation space is composed of linear combinations of feature vectors, and supervised attibution methods would likely struggle with successfully attributing non-linear features to their correct layer. (Is this true?) 

On a similar vein, both unsupervised and supervised methods treat layers and blocks in models as discrete modules where computation is taking place - SAEs reconstruct the MLP, residual steram, or attention head outputs from a single layer or block independently, and unsuperbised methods hope to identify a feature's causal layer or block. In reality,  deep neural networks are certainly not restricted to performing modular computations that can take place within a single block. Even in a transformer - an architecture relatively modular in design - we hypothesize that models utilize cross-layer and attention head superposition.  For architectures with even less separable structures - LSTMs, GANs, generative diffusion models, deep-Q networks, it seems even less straightforward which "activation space" is home to the the latent structures that encode relevant features. 

Finally, while working in activation space is helpful for model behavior at inference, it does little to help us understand the relationship between model behavior, training data, and the learning process. Understanding these relationships may be able to open up new avenues of intervention. Mapping undesirable behaviors to subsets of training data can help guide finetuning, and flagging critical points in the learning process might help us identify and steer problematic model capabilities early on. 

### Parameter Space Interpretability 

There is an alternative to interpreting models through their activations: interpreting models through their parameter space. Parameters are the fundamental entities updated during training, representing infromation learned by the interaction between the data and model's learning algorithm. Unlike the transient activation space seen during inference, the parameter space reflects the cumulative result of the training process, and may provide a more robust frame for understanding how models represent features, circuits and behaviors. 

There is already a growing body of work developing theories describing relationships between data, parameter space, and behavior. **Singular learning theory (SLT)** — a mathematical framework grounded in information geometry — aims to describe how the structure of parameter space influences generalization and model behavior (Watanabe, 2009). By modeling the training process as movement through a complex parameter manifold, SLT predicts how sharp minima, flat regions, and singularities in parameter space correspond to features and predictive power learned from data. SLT's more application-based descendent, **developmental interpretability**, extends these ideas by focusing on the trajectory of parameter updates during training. Developmental interpretability seeks to understand not just the final learned parameters, but also the intermediate representations that arise during training. SUch an approach has shown promise for understanding how models progressively encode features, circuits, and abstractions as training progresses, offering insights into both the learning dynamics and the eventual behavior of the model (Olah et al., 2020).

SLT, developmental interpretability, and other parameter-space analysis have already have already produced key insights into the relationship between learning, data and behavior. For example, various studies have shown that the sharpness of parameter space is at least somewhat responible for the phenomenon of grokking. transition aligns with the model moving into flatter regions of parameter space, which are associated with solutions that generalize better. In another study, in image classification tasks, certain parameters have been found to overfit rare examples, such as mislabeled or adversarial images, while others represent common patterns like edges or textures (Feldman, 2020). Identifying these "memorization-heavy" parameters can inform techniques like adversarial pruning, where parameters that overfit adversarial examples are selectively penalized or re-trained. Concepts from singular learning theory and developmental interpretabilty have used to connect sharp transitions in the loss landcape to learned features, quantify the loss geometry of in-context learning and other important stages, and propose more efficient architectures less prone to wasting parameter information capacity. 


### Sparse parameter decomposition 

In this work, we propose a method for identifying features and the circuits that produce them in large models using the parameter space. We propose a method that identifies directions in a model's parameter space - rather than activation space - that define specific circuits relevant for a small subsets of tasks.  Our method builds on the key idea in SLT that loss landscape geometry can help us understand model behavior, but also takes assumptions from sparse-autoencoder based approaches - namely that general purpose models rely on features and circuits that are used for only a subset of inputs and that sparsity contraints can expose such submodules. 

In short, we decompose the per-sample Hessian of a trained model with respect to parameter space, imposing sparsity contraints such that each direction in parameter-space is relevant for only a small subset of samples. In order to reduce computational complexity, we estimate the Hessian via the gradient of the loss between our model of interest and a uniform baseline model, and approximate directions in parameter space via lower-rank matrices. Noteably, it has some similarities to an underlooked method (HERE) - where the authors compute the principal directions of a per-sample Fisher Information matrix in order to resolve features. While they rely on diagonalization to reduce computational complexity and we use a gradient estimation and low rank matrices, the key motivations surrounding loss landscape geometry and sparse circuitry are quite similar. 

In this work, we first describe the mathematics behind our method, justifying key assumptions and approximations. Next, we showcase the results of our method, using both small toy models and real-world use case models. 

We first show that our method can perfectly resolve the feature landscape of two toy models:  
- An autoencoder popularly used as a toy model to showcase the phenomenon of superposition 
- A manifold 

We then show that our metohd outperforms SAEs in resolving features from two real-world use case models
- A small language model
- A convolutional neural network

Finally, we discuss the strengths and limitations of our method, and where it diverges from activation-space methods. The strengths are namely (1) the use of the paramter space instead of activations space, (2) ability to be used with architectures of any type, and (3) natural next steps to feature interaction maps. The methods weaknesses are primarily that (1) the method is restricted to a local region in the loss landscape space, which makes intervening less straightforward, and (2) per-sample gradients cannot be extracted in batches the same way per-sample activations used by SAEs can, which does not allow our method to take full advantage of parallel process of GPUs. 



## Methods

### Set up 

We begin by setting up the problem, introducing nomenclature, and defining terms. 

Our goal is to find computational **subnetworks of a parameterized network** - circuits that extract or compute various features. We hypothesize that such submodules will be characterized by directions in parameter space that strongly impact the output of a small subset of inputs, and trivially impact the outputs of others. 

**Let $f(x, w)$ be a the output of a parametric model**, where $x$ is an input vector and $w$ a parameter vector.  In this writeup, we examine several classes of models:

1. **Discriminatory regression models**, where the model predicts the value of an output vector $y$, based on the value of an input vector $x$. That is:

    $f(x, w) = {f_y(x, w)}_y $

2. **Discriminatory classification models, and generative models**, where the network models the probabilities of a finite number of classes $y$, based on the input vector $x$. That is:

    $f(x, w)$ = ${P(y \mid x, w)}_y$ where $ \sum_{y} P(y \mid x, w) = 1$ .


**Divergence** is a measurement for how widely two sets of predictions vary. In our methods, we always examine divergences between models that have the same output space and typically discuss divergence on a per-sample level. We rely on two different divergence metrics for our different classes of models:
 
1. For regression discriminatory type models, we use a squared-error loss. For a given input $x$, the divergence in the predictions between two models: $f$ (with parameters $w_{f}$),  and $g$ (with parameters $w_{g}$).

    $SE(f(x), g(x)) = \sum_y (f_y(x) - g_y(x))^2$

2. For discriminatory classification models, as well as generative models, we use the KL divergence to measure differences between probability distributions.

    $KL(f(x), g(x)) = \sum_y f_y(x) log \frac{f_y(x)}{g_y(x)}$


### Learning principle directions of the per-sample Hessian

We can represent the magnitude by which a given parameter, or pair of parameters affects a model's prediction by computing the **per-sample Hessian**, with respect to the parameters $w$ of the divergence ($D$) between a model's output and the output of an identical model with fixed parameters.  That is,

$H(f, x, w_0) = \nabla_{w}^{2} D(f(x, w), f(x,w_0))$ 

The divergence may be the SE or the KL divergence depending on model class. (Note that the *first-order gradient* of the divergence here will always be zero, because the divergence is at a global minima.)

Our goal is to find a set of directions $U$ in parameter space that fulfill two criteria:

**(1) The directions of $U$ can be used to approximately construct the per-sample hessian of a given model, for all samples in a specific dataset $\{x\}$**

$
H' = U^T U H(f, x) U^T U
$

$
H' \approx H
$

**(2) For any single input, we'd also like a small smaller subset of $U$ to be able to approximately reconstruct the per-sample Hessian.** Put another way, for many rows of $U$, their contribution to the Hessian reconstruction should be zero. 

For $x \in X$:

$U' H(f, x) U'^T \approx 0 $ where $U' \in U$


Let's briefly review the dimenions of the tensors we are working with:
- The per-sample Hessian $H$ will be of dimensions $|W|$ x $|W|$ where $|W|$ is the number of parameters in the model.  We may have $n$ per-sample Hessians to work with, where $n$ is the number of samples in the dataset of interest.
- $U$ will be a matrix of $n_{\text{vectors}}$ x $|W|$ where $n_{\text{vectors}}$ will be a hyperparameter describing the number of vectors we wish to decompose the Hessian into. 

### Approximating the Hessian with a first-order gradient

Computing and decomposing the second-order gradient (Hessian) of a model for every sample would be extremely computationally expensive. We propose to instead compute and decompose the first-order gradients of the a loss between the model of interest and a baseline model, designed to rely on the same principle vectors as the Hessian. We explain why these approximately spanning sets - from here on called "principal vectors" of this decomposition will be reasonable approximations for those of the original Hessian.  

Let's first compute what the functional form of the Hessian looks like for our two types of divergences - squared error and KL-divergence.

For SE, the per-sample Hessian is:

$$
\nabla^{2}_{w}SE(f, x) = (\nabla_{w} f(x))^T \nabla_{w} f(x)
$$

For KL-divergence, the per-sample Hessian is:

$$
\nabla^{2}_{w}KL(f, x) = (\nabla_{w} f(x))^T \sigma \nabla_{w} f(x)
$$


Now, consider two new divergences ($D'$):

(1) The difference between a outputs of a regression model, and a baseline model with constant outputs. 

$
SE' = \sum_i f_i(x, w)-c
$

(2) the KL divergence of a baseline model that outputs uniform probabilities $p=n^{-1}$, where $n$ is the number of available classes compared to our model of interest.

$
KL' = \sum_i p \text{ log} \frac{p}{f_i(x)}
$


The first order gradients of these divergences will then be the following. For SE:

$
\nabla SE'(f,x) = (\nabla f(x, w))^T 1_{|f(x,w)|}
$

where  $1_{|f(x,w)|}$ is a vector of 1s the same size as the output for $f(x,w)$.

For KL-divergence:

$
\nabla KL'(f,x) = (\nabla f(x, w))^T \sigma' 
$

where $\sigma'$ is a vector where $\sigma'_{i}=f_i(x, w)$.

Our sparse principal vectors for these first-order gradients of divergence will be good approximations of the sparse principal vectors of the previously discussed second-order gradients. Namely, they will satisfy the following criteria.

***Proposition: If a set of vectors $U$ spans the a sample's first-order gradients, the same set of vectors can reconstruct the sample's Hessian.***

$$
\nabla_w KL'(f,x) = \nabla KL'(f,x) u = (\nabla f(x, w))^T \sigma' 
$$

***Proposition: If the first-order gradient in direction $u$ is zero, the per-sample second-order gradient will also be zero.***


### Training

Our goal is to learn principle parameter vectors ($U$) that best satisfy our reconstruction criteria and sparsity criteria. 

$
L_{\text{reconstruct}}(x) = - (\nabla D'(f,x) - \nabla D'(f,x) U^T U)^2
$

We use an $L_1$ loss to encourage sparsity. 
$
L_{\text{sparsity}}(x) = ||\nabla D'(f,x) U^T||
$

$
L(x) = L_{\text{reconstruct}}(x) + \lambda L_{\text{sparsity}}(x)
$

    For each epoch:
        For x_batch in X:
            L = - sum([L_reconstruct(x) for x in x_batch]) + lambda sum([L_sparsity(x) for x in x_batch])
            Backward pass on L
            Update U
            



### Low-rank approximations of parameter directions

We have described how we can find sparse principal vectors of a model's Hessian by finding sparse princniple vectors of a the first order gradient of a model's divergence with a simple baseline model. Working with first-order gradient greatly reduces the computational complexity of identifying such vectors, since computing the first order-gradient of a sample requires one backwards pass, while computing the Hessian of a sample requires $|W|$, and $|W|$ can be many orders of magnitude for large models. 

Similarly, learning $U$ where each row is length $|W|$ is memory- and computationally-intensive. Instad, each $u \in U$ will be a low-rank decomposition of tensors, similar to those used in LoRA (reference). 

Let $\{w_j\}_k$ represent the collection of model's parameter tensors. 

Each $u_i \in U$  can then be written as a flattened version of 

$
u_{i,j}  =  \sum_{k=1}^r A_{i,j,k} \otimes B_{i,j,k} \otimes C_{i,j,k} ...
$ 


where 

- $ A_{i,j,k} \in \mathbb{R}^a $, $B_{i,j,k} \in \mathbb{R}^b $, $C_{i,j,k} \in \mathbb{R}^c $...
- $r$ is the rank of the decomposition.
- $\text{dim}(w_j) = (a, b, c...)$

$u_{i}$ will then be the set of low-rank representations for each tensor in the parameter set. 

$
u_i = \{u_{i,j}\}_j
$

Therefore, instead of learning a parameters vector with $|W|$ dimensions for each principal vector, we will learn each principal vector's low-rank factors instead: $\{A_{i,j,k}, B_{i,j,k}, C_{i,j,k}\}$.


## Results


### Toy model of superposition

#### Set up 


#### Decomposition 



### Non-linear manifold

#### Set up 


#### Decomposition 


### Convolutional Neural Network

#### Set up 

#### Decomposition

#### Comparison 

### Transformer 


## Discussion 







$\mathcal{D}(w) = \sum_{y=1}^K P(y \mid x,w) \log \frac{P(y \mid x,w)}{P(y \mid x,w')}.
   $
   At $w = w'$, this divergence vanishes ($\mathcal{D}(w')=0$). The Hessian at $w'$ is known to coincide with the Fisher Information at $w'$:
   $
   \nabla_w^2 \mathcal{D}(w') = I(w'),
   $
   where
   $
   I(w') = \sum_{y=1}^K P(y \mid x,w') G_y(w') G_y(w')^\top.
   $

   Thus, the Hessian $\nabla_w^2 \mathcal{D}(w')$ is a sum of outer products of the gradient vectors $G_y(w')$.


 terms inside the sum can be expressed using $k_y(x, w')$. Therefore, there are scalars $c_y(w')$ such that



$
\nabla_w \mathcal{L}(w') = \sum_y c_y(w') L_y(w').
$
   
   This shows that $\nabla_w \mathcal{L}(w')$ is a linear combination of the vectors $G_y(w')$.

2. **Representation of $\nabla_w^2 \mathcal{D}(w')$:**  
   Consider now
   $
   \mathcal{D}(w) = \sum_{y=1}^K P(y \mid x,w) \log \frac{P(y \mid x,w)}{P(y \mid x,w')}.
   $
   At $w = w'$, this divergence vanishes ($\mathcal{D}(w')=0$). The Hessian at $w'$ is known to coincide with the Fisher Information at $w'$:
   $
   \nabla_w^2 \mathcal{D}(w') = I(w'),
   $
   where
   $
   I(w') = \sum_{y=1}^K P(y \mid x,w') G_y(w') G_y(w')^\top.
   $

   Thus, the Hessian $\nabla_w^2 \mathcal{D}(w')$ is a sum of outer products of the gradient vectors $G_y(w')$.

3. **Common Set of Vectors:**
   From Steps 1 and 2, we have:
   - $\nabla_w \mathcal{L}(w')$ is a linear combination of $\{G_y(w')\}$.
   - $\nabla_w^2 \mathcal{D}(w')$ is a linear combination of the outer products $\{G_y(w')G_y(w')^\top\}$.

   Since both the gradient $\nabla_w \mathcal{L}(w')$ and the Hessian $\nabla_w^2 \mathcal{D}(w')$ can be constructed entirely from the set $\{G_y(w')\}$, it follows that this set of vectors suffices to represent both first-order information about the divergence $\mathcal{L}(w)$ (with respect to a fixed $Q$) and second-order information about the divergence $\mathcal{D}(w)$ (with respect to $w'$).

**Conclusion:**
The family of vectors $\{G_y(w')\}_{y=1}^K$, which encapsulate the local sensitivities of the log-probabilities under $P$, provides a common basis from which both the Jacobian $\nabla_w \mathcal{L}(w')$ and the Hessian $\nabla_w^2 \mathcal{D}(w')$ can be derived. This establishes that the same fundamental building blocks in parameter space underlie both the first-order and second-order structures of these divergences.






, and an isotropic baseline model, is a good approximation of the basis for the Hessian above. 



$U \text{ spans } H(f, {x}, w_0)$

For $x \in {x}$:

$U' \in U \text{ spans } H(f, x, w_0)$ where $|U'| << |U|$





Below is a proof outline that establishes that the same family of gradient vectors used to form the Jacobian (of a divergence between $ P $ and a fixed $ Q $) can also be used to represent the Hessian (of a divergence between $ P $ and $ P $) with respect to the parameters $ w $.


**Setup:**

1. **Models and Divergences:**
   Consider a parametric model $ P(y \mid x, w) $ over a finite label set $\{1,\ldots,K\}$. We are interested in two divergences:

   - $\mathcal{L}(w) = \text{KL}(P(\cdot \mid x,w) \| Q(\cdot \mid x))$, where $ Q $ does not depend on $ w $.
   - $\mathcal{D}(w) = \text{KL}(P(\cdot \mid x,w) \| P(\cdot \mid x,w')) $ for some fixed $ w' $.

   The gradient (Jacobian) of $\mathcal{L}(w)$ with respect to $ w $ and the Hessian of $\mathcal{D}(w)$ at $ w' $ are the objects of interest.

2. **Gradient Vectors:**
   Define for each class $ y $:
   $
   G_y(w') := \nabla_w \log P(y \mid x,w').
   $
   These vectors $ G_y(w') \in \mathbb{R}^d $ (where $ d = \dim(w) $) represent the sensitivity of the log-probability of class $ y $ to changes in $ w $ at the point $ w' $.

**Part 1: Representing the Jacobian Using $\{G_y(w')\}$**

The gradient of $\mathcal{L}(w)$ at $ w' $ can be expressed in terms of $ G_y(w') $. Since:
$
\mathcal{L}(w) = \sum_{y=1}^K P(y \mid x,w) \log\frac{P(y \mid x,w)}{Q(y \mid x)},
$
its gradient at $ w' $ is:
$
\nabla_w \mathcal{L}(w') = \sum_{y=1}^K A_y(w') G_y(w'),
$
for some scalars $A_y(w')$ that depend on $ P(\cdot \mid x,w') $ and $ \log Q(\cdot \mid x) $. Crucially, $\nabla_w \mathcal{L}(w')$ is a linear combination of the vectors $\{G_y(w')\}$. Thus, the Jacobian of $\mathcal{L}$ at $ w' $ lies in the span of $\{G_y(w') : 1 \le y \le K\}$.

**Part 2: Representing the Hessian Using $\{G_y(w')\}$**

Now consider $\mathcal{D}(w) = \text{KL}(P(\cdot \mid x,w) \| P(\cdot \mid x,w'))$. By definition:
$
\mathcal{D}(w) = \sum_{y=1}^K P(y \mid x,w) \log\frac{P(y \mid x,w)}{P(y \mid x,w')}.
$

At $ w = w' $, $\mathcal{D}(w') = 0$. The second derivative (Hessian) of $\mathcal{D}(w)$ at $ w' $ is known to coincide with the Fisher Information at $ w' $:
$
\nabla_w^2 \mathcal{D}(w') = I(w'),
$
where
$
I(w') = \sum_{y=1}^K P(y \mid x,w') G_y(w') G_y(w')^\top.
$

This shows that the Hessian is a sum of outer products $ G_y(w') G_y(w')^\top $, each of which is formed from vectors in the set $\{G_y(w')\}$. In other words, $ I(w') $, and hence $\nabla_w^2 \mathcal{D}(w')$, lies in the space of matrices spanned by the outer products of these gradient vectors.

**Part 3: The Same Set of Vectors for Both Jacobian and Hessian**

- For $\nabla_w \mathcal{L}(w')$: We have a linear combination of $\{G_y(w')\}$.
- For $\nabla_w^2 \mathcal{D}(w')$: We have a linear combination of $\{G_y(w')G_y(w')^\top\}$.

Since every $ G_y(w')G_y(w')^\top $ involves only vectors from the set $\{G_y(w')\}$, and every $ G_y(w') $ obviously comes from the same set, it follows that the vector space generated by $\{G_y(w')\}$ is sufficient to express both the gradient of the $ P \| Q $ divergence and the Hessian of the $ P \| P $ divergence at $ w' $.

Concretely, if you know the vectors $ \{G_y(w')\} $, you can form any linear combination to get $\nabla_w \mathcal{L}(w')$. Also, with these same vectors you can form their outer products and sum them up (with appropriate weights) to get $\nabla_w^2 \mathcal{D}(w')$.

**Conclusion:**

The key insight is that both the Jacobian of $\mathcal{L}(w)$ (the divergence between $P$ and a fixed $Q$) and the Hessian of $\mathcal{D}(w)$ (the divergence between $ P $ and $ P $) are built from the same fundamental building blocks: the parameter gradients $ G_y(w') $. Since these gradients define both the linear structure needed to represent the Jacobian and the quadratic structure (via outer products) needed to represent the Hessian, the same set of vectors $\{G_y(w')\}$ is sufficient to compose both the Jacobian and the Hessian under these conditions.



**Proof:**

1. **Representation of $\nabla_w \mathcal{L}(w')$:**  
   By the definition of the KL divergence,
   $
   \mathcal{L}(w) = \sum_{y=1}^K P(y \mid x,w) \log \frac{P(y \mid x,w)}{Q(y \mid x)}.
   $
   Differentiating w.r.t. $w$ and evaluating at $w'$,
   $
   \nabla_w \mathcal{L}(w') = \sum_{y=1}^K \left[ \nabla_w P(y \mid x, w') \log \frac{P(y \mid x, w')}{Q(y \mid x)} \;+\; P(y \mid x, w') \nabla_w \log P(y \mid x,w') \right].
   $

   Since $\nabla_w P(y \mid x, w') = P(y \mid x, w') G_y(w')$, both terms inside the sum can be expressed using $G_y(w')$. Thus, there exist scalars $\{A_y(w')\}_{y=1}^K$ such that
   $
   \nabla_w \mathcal{L}(w') = \sum_{y=1}^K A_y(w') G_y(w').
   $
   
   This shows that $\nabla_w \mathcal{L}(w')$ is a linear combination of the vectors $G_y(w')$.

2. **Representation of $\nabla_w^2 \mathcal{D}(w')$:**  
   Consider now
   $
   \mathcal{D}(w) = \sum_{y=1}^K P(y \mid x,w) \log \frac{P(y \mid x,w)}{P(y \mid x,w')}.
   $
   At $w = w'$, this divergence vanishes ($\mathcal{D}(w')=0$). The Hessian at $w'$ is known to coincide with the Fisher Information at $w'$:
   $
   \nabla_w^2 \mathcal{D}(w') = I(w'),
   $
   where
   $
   I(w') = \sum_{y=1}^K P(y \mid x,w') G_y(w') G_y(w')^\top.
   $

   Thus, the Hessian $\nabla_w^2 \mathcal{D}(w')$ is a sum of outer products of the gradient vectors $G_y(w')$.

3. **Common Set of Vectors:**
   From Steps 1 and 2, we have:
   - $\nabla_w \mathcal{L}(w')$ is a linear combination of $\{G_y(w')\}$.
   - $\nabla_w^2 \mathcal{D}(w')$ is a linear combination of the outer products $\{G_y(w')G_y(w')^\top\}$.

   Since both the gradient $\nabla_w \mathcal{L}(w')$ and the Hessian $\nabla_w^2 \mathcal{D}(w')$ can be constructed entirely from the set $\{G_y(w')\}$, it follows that this set of vectors suffices to represent both first-order information about the divergence $\mathcal{L}(w)$ (with respect to a fixed $Q$) and second-order information about the divergence $\mathcal{D}(w)$ (with respect to $w'$).

**Conclusion:**
The family of vectors $\{G_y(w')\}_{y=1}^K$, which encapsulate the local sensitivities of the log-probabilities under $P$, provides a common basis from which both the Jacobian $\nabla_w \mathcal{L}(w')$ and the Hessian $\nabla_w^2 \mathcal{D}(w')$ can be derived. This establishes that the same fundamental building blocks in parameter space underlie both the first-order and second-order structures of these divergences.


## Approximating the Hessian with first-order gradient

## Low-rank matrices

## Sparse eigenbasis training



## Toy model of superposition 

## Transformer

##




### Model
The output of model $f$ (of a given architecture), depends on two inputs - a set of parameters $W$ and a set of feature $x$.

After training a given model, we end up with a set of weights/parameters $W_0$.


### Divergence
The divergence metric (not called loss, because I use that term later) should describe how much a model's output changes (for a given sample) when it's parameters are altered. 


In the case of a transformer, we can compute the normalized KL-divergence. The KL-divergence describes the divergence between probability distributions and we can normalize this metric to have a minimum of zero using the following equation (note that the unnormalized KL divergence does not necessarily have a minimum at 0). We could potentially use a normalized cross entropy here as well. 

$
D(f, x, W) = KL(f(W, x), f(W_0, x)) - KL(f(W_0, x), f(W_0, x))
$

In the case of a regression problem, we can use the mean squared error loss as our divergence metric. We don't need to normalize it because MSE does have a guarenteed minimum at 0. 

$
D(f, x, W) = MSE(f(W, x), f(W_0, x))
$

### Sample-level Hessian 

We define the sample-level hessian as the second derivative of a model's loss with respect a set of parameters $w$, evaluated at $W0$ and X. 

$
H(x,w) = \nabla^{2}_{w} D(f, x, W)
$

Using the Hessian, we can compute the second derivative of the loss with respec to any direction in weight/parameter space (given by vector $u$)  by:

$
 \nabla^{2}_{u} D(f, x, W) = u(u H(x,w))^T
$

### Learning feature vectors

Our goal is to identify important directions (${u}$ or $U$) in parameter space of this Hessian where:
(1) For some samples, $\nabla^{2}_{u} D$ is very high. Moving parameters in the direction of $u$ dramatically changes the output of the model. 
(2) There is low sample-level interference between each $u$ vector. $u$ vectors should either be nearly orthogonal, or if they have high cosine similarities, they should not both have high $\nabla^{2}_{u} D$ for a given sample. 

We wish to learn a set of u-vectors ($U$) that satisfy these conditions. We will therefore minimize two losses. 

To satisfy the first goal we minimize:

$
L_{\text{steep hessian}}(x) = \sum_{u}(\nabla^{2}_{u} D(f, x, W_0))^2
$

To satisfy the second goal, we also want to minimize:

$
L_{\text{low interference}}(x) = \|\nabla^{2}_{u1}D \space u_1 (\nabla^{2}_{u2}D \space u_2)^{T} \|_{u_1, u_2}
$


### Optimization tricks 
Optimizing losses that depend on a full hessian is computationally expensive. We use two tricks to minimize these losses.

#### Jacobian-vector products
First, Instead of forming the full hessian, or full jacobian, we can compute nested jacobian-vector products.

$
\nabla^{2}_{u} D(f, X, W_0) = \nabla_{w} (\nabla_{w} D(f, X, W_0) \space u ) \space u
$


#### Bi-level optimization

We use two separate optimization loops and perform bi-level optimization. 

    For x_batch in X:
        L1 = 0
        d2D_all = []

        # First optimization step.

        For u_batch in U:
        
            # Compute steep hessian loss.
            1. d2D(U) = second derivative of D(f, x, W_0) with respect to u_batch.
            2. L1 = L1 + sum(d2D^2) # Compute first loss. 
            3. d2D_all = d2D_all.append(d2D(U))
        
        
        U = U + L1*step    # Update U

        # Second optimization step. 
        L2 = || (U d2D_all) (U d2D_all)^T  || # Compute low-interference loss. 
        U = U + (L1 + \lambda L2)*step # Update U.


Note that in our first optimization loop (minimizing $L_{\text{low itnerference}}$), we can compute L in batches of U.


## Toy models

### XOR Model

We train a VERY simple neural network to learn the "XOR" function. The NN network consists of a single hidden layer of 2 nodes, with Gelu activation functions. The output of the hidden layer is summed to get a final output. There are 4 parameters today (2 weights an 2 biases). The training data for this network looks like:

    [0, 1] --> 1
    [1, 0] --> 1
    [0, 0] --> 0
    [1, 1] --> 0

### Toy model's of superposition 

We train a TMS (autoencoder 5 features, 2 hidden dimensions, Relu activation, with W_in and W_out as transposes). We set the features to uniform random numbers bewteen 0 and 1, with 5% sparsity. The TMS model successfully represents the features in pentagonal superposition.  

### Transformers

We use the tiny-stories-1M transformer.

## Results


### XOR Model

The eigenmodel successfully finds features that are most highly activated by [0,1], [1,1], and [1,0].

    feature_idx
    [sample input values] -> feature value

    feature 0
    [1. 1.] -> 2.3252432
    [1. 0.] -> 1.0929958
    [0. 1.] -> 0.61902356
    [0. 0.] -> 0.273664

    feature 1
    [0. 1.] -> 2.670694
    [1. 0.] -> 0.65577537
    [1. 1.] -> 0.52892005
    [0. 0.] -> 0.15629882

    feature 2
    [1. 0.] -> 2.021866
    [1. 1.] -> 0.39354768
    [0. 1.] -> 0.035297774
    [0. 0.] -> 0.011965705


### TMS

The most highly activating samples are the following:

    feature_idx
    [sample input values] -> feature value

    feature 0
    [0.    0.779 0.    0.    0.948] -> 3.342
    [0.    0.833 0.    0.    0.35 ] -> 2.111
    [0.    0.999 0.    0.    0.   ] -> 1.862
    [0.    0.997 0.    0.    0.   ] -> 1.857
    [0.    0.272 0.    0.    0.944] -> 1.836

    feature 1
    [0.888 0.    0.    0.835 0.   ] -> 3.406
    [0.795 0.    0.    0.812 0.   ] -> 3.075
    [0.796 0.    0.    0.78  0.   ] -> 2.977
    [0.472 0.    0.    0.881 0.   ] -> 2.455
    [0.285 0.    0.    0.91  0.   ] -> 2.101

    feature 2
    [0.758 0.    0.    0.    0.975] -> 2.078
    [0.641 0.    0.    0.    0.698] -> 1.383
    [0.517 0.    0.    0.    0.744] -> 1.295
    [0.    0.    0.    0.    0.999] -> 1.274
    [0.    0.    0.    0.    0.992] -> 1.259

    feature 3
    [0.758 0.    0.    0.    0.975] -> 2.516
    [0.641 0.    0.    0.    0.698] -> 1.724
    [0.979 0.    0.    0.    0.   ] -> 1.666
    [0.971 0.    0.    0.    0.   ] -> 1.646
    [0.517 0.    0.    0.    0.744] -> 1.546

    feature 4
    [0.126 0.    0.942 0.477 0.   ] -> 2.439
    [0.    0.    0.984 0.    0.   ] -> 2.241
    [0.    0.    0.971 0.    0.   ] -> 2.199
    [0.   0.   0.97 0.   0.  ] -> 2.196
    [0.    0.    0.967 0.    0.   ] -> 2.187

If we only consider completely sparse samples, the top features are the following. 


    feature_idx
    [sample input values] -> feature value

    feature 0
    [0.    0.999 0.    0.    0.   ] -> 1.862
    [0.    0.997 0.    0.    0.   ] -> 1.857
    [0.    0.981 0.    0.    0.   ] -> 1.816
    [0.    0.971 0.    0.    0.   ] -> 1.793
    [0.    0.957 0.    0.    0.   ] -> 1.758
    
    feature 1
    [0.    0.    0.    0.998 0.   ] -> 1.728
    [0.    0.    0.    0.996 0.   ] -> 1.724
    [0.    0.    0.    0.995 0.   ] -> 1.721
    [0.    0.    0.    0.991 0.   ] -> 1.712
    [0.   0.   0.   0.97 0.  ] -> 1.662
    
    feature 2
    [0.    0.    0.    0.    0.999] -> 1.274
    [0.    0.    0.    0.    0.992] -> 1.259
    [0.    0.    0.    0.    0.969] -> 1.213
    [0.    0.    0.    0.    0.952] -> 1.181
    [0.    0.    0.    0.    0.938] -> 1.154

    feature 3
    [0.979 0.    0.    0.    0.   ] -> 1.666
    [0.971 0.    0.    0.    0.   ] -> 1.646
    [0.915 0.    0.    0.    0.   ] -> 1.523
    [0.901 0.    0.    0.    0.   ] -> 1.493
    [0.897 0.    0.    0.    0.   ] -> 1.485
    
    feature 4
    [0.    0.    0.984 0.    0.   ] -> 2.241
    [0.    0.    0.971 0.    0.   ] -> 2.199
    [0.   0.   0.97 0.   0.  ] -> 2.196
    [0.    0.    0.967 0.    0.   ] -> 2.187
    [0.    0.    0.958 0.    0.   ] -> 2.157


### Transformer
Here are some results using just the weights in transformer.blocks.4.attn.W_K (4096 weights) in the tiny-stories-1M model.  Here are some results from a model trained very briefly (<10 epochs, on only 500 token sets) with only 10 features pulled out. Results on data the eigenmodel was not trained on.

    

    Feature description (by me)
    tokens (activating token bolded) -> Feature value

    Word/punctuation after a name
    upon a time, there was a kind man named Tom**.** Tom had a big -> . (Value: 42.845)
    zoo.Once upon a time, there was a little boy named Timmy**.** -> . (Value: 40.725)
    upon a time, there was a woman named Lily.** She** loved to go for ->  She (Value: 38.833)
    Once upon a time, there was a little boy named Timmy**.** Timmy -> . (Value: 38.542)
    Once upon a time, there was a little boy named Timmy**.** Timmy -> . (Value: 38.542)

    Once upon a time
    upon a time, there was a woman named Lily.** She** loved to go for ->  She (Value: 64.518)
    could always ask the black cat.Once upon a time**,** there was a small -> , (Value: 56.562)
    loved ones. The end.Once upon a time**,** there was a little girl -> , (Value: 50.217)
    enjoyed the sunshine.Once upon a time**,** there was a little boy -> , (Value: 46.414)
    street and always looked out for banana peels.Once upon a time**,** there -> , (Value: 46.287)

    Tim
    and trucks. One day,** Tim**my's dad took him to the park to ->  Tim (Value: 69.034)
    lost and felt sad.newlinenewlineOne day,** Tim**my's friend, a ->  Tim (Value: 66.466)
    toys. The next day,** Tim**my went to his friend's house and said ->  Tim (Value: 62.213)
    was playing just as well as before.** Tim**my was so happy with his new ->  Tim (Value: 57.687)
    the noisy trunk.Once upon a time, there was a boy named** Tim**my ->  Tim (Value: 54.684)

    ?
    park. One day, she saw a big statue of a dog**.** It was -> . (Value: 30.722)
    't like that. He wanted his truck to be the** best**. They argued for ->  best (Value: 27.736)
    every day. One day, he saw a little bird on a branch**.** The -> . (Value: 26.222)
    newlinenewlineOne day, Timmy saw a black cat in his backyard**.** The -> . (Value: 22.828)
    ily said, "Let's ask for help!" They saw a man** and** asked ->  and (Value: 22.407)

    Gender / polysemanic
    upon a time, there was a woman named Lily.** She** loved to go for ->  She (Value: 59.775)
    Once upon a time, there was an old man.** He** liked to read magazines ->  He (Value: 39.421)
    the noisy trunk.Once upon a time, there was a boy named** Tim**my ->  Tim (Value: 38.685)
    summer.Once upon a time, there was an old lady.** She** was very ->  She (Value: 37.195)
    came in to see what was going on.newlinenewlineShe told** Tim**my that ->  Tim (Value: 36.652)

    saw + polysemantic
    the park every day. newlinenewlineOne day, the man** saw** the woman ->  saw (Value: 39.632)
    the dark hole in the ground and he fell in.newlinenewline**Tim**my tried -> Tim (Value: 30.418)
    and went inside the tent. newlinenewlineThe lion** saw** a man with a ->  saw (Value: 28.665)
    Allowed." Timmy was sad because he wanted Max to come with them**.** -> . (Value: 28.455)
    's mommy lifted her up so she could see over it. She** saw** her ->  saw (Value: 28.105)

    Tim + "then on?"
    toy tools. From that day on,** Tim**my learned the importance of sharing and ->  Tim (Value: 75.689)
    was playing just as well as before.** Tim**my was so happy with his new ->  Tim (Value: 63.231)
    at them. He even got to touch a baby shark! After that,** Tim** ->  Tim (Value: 59.772)
    then, his mom came in and asked what was wrong.** Tim**my told her ->  Tim (Value: 58.194)
    the dark hole in the ground and he fell in.newlinenewline**Tim**my tried -> Tim (Value: 57.360)


    Animals?
    named Timmy. He had a big, brown dog named Max**.** Max loved -> . (Value: 54.549)
    park. One day, she saw a big statue of a dog**.** It was -> . (Value: 43.561)
    She loved to walk on the trail with her dog, Max**.** Max was very -> . (Value: 39.359)
    newlinenewlineOne day, Timmy saw a black cat in his backyard**.** The -> . (Value: 36.026)
    and wanted to make him feel better. Tommy** told** Sammy a joke and Sammy laughed ->  told (Value: 34.926)


    was playing just as well as before.** Tim**my was so happy with his new ->  Tim (Value: 55.711)
    part of the park where dogs were allowed.** Tim**my was happy again because he ->  Tim (Value: 38.506)
    then, his mom came in and asked what was wrong.** Tim**my told her ->  Tim (Value: 37.558)
    should answer with courage. So,** Tim**my took a deep breath and stood up ->  Tim (Value: 34.767)
    came in to see what was going on.newlinenewlineShe told** Tim**my that ->  Tim (Value: 33.680)


    came in to see what was going on.newlinenewlineShe told** Tim**my that ->  Tim (Value: 46.779)
    at them. He even got to touch a baby shark! After that,** Tim** ->  Tim (Value: 34.167)
    the dark hole in the ground and he fell in.newlinenewline**Tim**my tried -> Tim (Value: 30.990)
    part of the park where dogs were allowed.** Tim**my was happy again because he ->  Tim (Value: 30.866)
    that they were safe.newlinenewlineBut the next day,** Tim**my didn't ->  Tim (Value: 30.029)